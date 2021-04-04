import torch
from torch import nn
from torch.nn import functional as F

from typing import Dict, Optional
from utils.misc_utils import combine_first_ax
from slowfast.models.video_model_builder import SlowFast, ResNet
from fairseq.models.transformer import (
    TransformerEncoder,
    TransformerDecoder,
    # EncoderOut,
)
from utils.transformer_code import Transformer as TxCodeEnc

from vidsitu_code.seq_gen import SeqGenCustom, EncoderOut
from transformers import GPT2LMHeadModel
from vidsitu_code.hf_gpt2_fseq import HuggingFaceGPT2Decoder


class SlowFast_FeatModel(SlowFast):
    def forward_features(self, x):
        x = self.s1(x)
        x = self.s1_fuse(x)
        x = self.s2(x)
        x = self.s2_fuse(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s3_fuse(x)
        x = self.s4(x)
        x = self.s4_fuse(x)
        x = self.s5(x)
        return x

    def forward(self, x, bboxes=None):
        x = self.forward_features
        if self.enable_detection:
            x = self.head(x, bboxes)
        else:
            x = self.head(x)
        return x


class ResNet_FeatModel(ResNet):
    def forward_features(self, x):
        x = self.s1(x)
        x = self.s2(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s4(x)
        x = self.s5(x)
        return x

    def forward(self, x, bboxes=None):
        if self.enable_detection:
            x = self.head(x, bboxes)
        else:
            x = self.head(x)
        return x


class ResNetBasicHead_Trimmed(nn.Module):
    """
    ResNe(X)t 3D head.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1x1. If the inputs are from multiple
    different pathways, the inputs will be concatenated after pooling.
    """

    def __init__(self, dim_in, pool_size):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetBasicHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
        """
        super().__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)
        self.dim_in = dim_in
        for pathway in range(self.num_pathways):
            if pool_size[pathway] is None:
                avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            else:
                # avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
                avg_pool = nn.AvgPool3d(pool_size[pathway])
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

    def forward(self, inputs):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(inputs[pathway]))
        x = torch.cat(pool_out, 1)

        return x


class SFBase(nn.Module):
    def __init__(self, cfg, comm):
        super(SFBase, self).__init__()
        self.full_cfg = cfg
        self.sf_cfg = cfg.sf_mdl
        self.cfg = cfg.mdl
        self.comm = comm
        self.build_model()

    def build_model(self):
        self.build_sf_model(self.sf_cfg)
        self.build_head(self.sf_cfg)
        self.build_projection_head(self.sf_cfg)

    def build_sf_model(self, cfg):
        mdl_name = cfg.MODEL.MODEL_NAME
        if mdl_name == "SlowFast":
            mdl = SlowFast_FeatModel(cfg)
        elif mdl_name == "ResNet":
            mdl = ResNet_FeatModel(cfg)
        else:
            raise NotImplementedError

        self.sf_mdl = mdl
        return

    def build_head(self, cfg):
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        # pool_size = _POOL1[cfg.MODEL.ARCH]
        if self.comm.path_type == "multi":
            self.head = ResNetBasicHead_Trimmed(
                dim_in=[
                    width_per_group * 32,
                    width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
                ],
                pool_size=[None, None],  # None for AdaptiveAvgPool3d((1, 1, 1))
            )
        elif self.comm.path_type == "single":
            self.head = ResNetBasicHead_Trimmed(
                dim_in=[width_per_group * 32],
                pool_size=[None],  # None for AdaptiveAvgPool3d((1, 1, 1))
            )

        return

    def build_projection_head(self, cfg, out_dim=None):
        if out_dim is None:
            out_dim = len(self.comm.vb_id_vocab)
        din = sum(self.head.dim_in)
        self.proj_head = nn.Sequential(
            *[nn.Linear(din, din // 2), nn.ReLU(), nn.Linear(din // 2, out_dim)]
        )

    def get_feats(self, inp):
        if self.comm.path_type == "multi":
            feat_slow = combine_first_ax(inp["frms_ev_slow_tensor"])
            feat_fast = combine_first_ax(inp["frms_ev_fast_tensor"])
            feats_used = [feat_slow, feat_fast]
        elif self.comm.path_type == "single":
            feat_fast = combine_first_ax(inp["frms_ev_fast_tensor"])
            feats_used = [feat_fast]
        else:
            raise NotImplementedError

        return feats_used

    def forward_encoder(self, inp):
        feats_used = self.get_feats(inp)
        nfeats_used = len(feats_used)
        feat_out = self.sf_mdl.forward_features(feats_used)
        assert len(feat_out) == nfeats_used
        return feat_out

    def forward_decoder(self, enc_out, inp):
        # enc_out: List
        # len(enc_out) = nfeats_used
        # enc_out[0]: B x C x T x H x W
        head_out = self.head(enc_out)
        # (B, C, T, H, W) -> (B, T, H, W, C).
        head_out = head_out.permute((0, 2, 3, 4, 1))

        # B = len(inp["vseg_idx"])
        # assert head_out.size(1) == 1
        # assert head_out.size(2) == 1
        # assert head_out.size(3) == 1

        # out = head_out.view(B, 5, -1)
        # import pdb

        # pdb.set_trace()

        proj_out = self.proj_head(head_out)
        B = len(inp["vseg_idx"])
        out = proj_out.view(B, 5, -1)
        assert out.size(-1) == len(self.comm.vb_id_vocab)
        return out

    def forward(self, inp: Dict):
        feat_out = self.forward_encoder(inp)
        mdl_out = self.forward_decoder(feat_out, inp)
        return {"mdl_out": mdl_out}


class LossB(nn.Module):
    def __init__(self, cfg, comm):
        super().__init__()
        self.cfg = cfg
        self.comm = comm
        self.loss_keys = ["loss"]

    def forward(self, mdl_out, inp):
        labels_c1 = combine_first_ax(inp["label_tensor"])
        mdl_preds = mdl_out["mdl_out"]
        mdl_preds_c1 = combine_first_ax(mdl_preds)
        loss = F.cross_entropy(mdl_preds_c1, labels_c1)
        return {"loss": loss}


class LossLambda(nn.Module):
    def __init__(self, cfg, comm):
        super().__init__()
        self.cfg = cfg
        self.comm = comm
        self.loss_keys = ["loss"]

    def forward(self, mdl_out, inp):
        assert "loss" in mdl_out
        return {"loss": mdl_out["loss"]}


class TxEncoderOld(TransformerEncoder):
    def __init__(self, cfg, comm):
        self.full_cfg = cfg
        self.comm = comm
        # dictionary = comm.vb_id_vocab
        dct_id = comm.dct_id
        dictionary = comm[dct_id]
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad_token_id
        args = cfg.tx_dec
        embed_dim = args.encoder_embed_dim
        embed_toks = nn.Embedding(num_embeddings, embed_dim, padding_idx)

        super().__init__(args, dictionary, embed_toks)
        self.after_init()

    def after_init(self):
        return

    def forward_embedding(
        self, src_tokens, token_embedding: Optional[torch.Tensor] = None
    ):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed

    def forward(
        self,
        src_tokens,
        src_lengths,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)

        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None,
        )


class TxEncoderNew(TxCodeEnc):
    def __init__(self, cfg, comm):
        self.full_cfg = cfg
        self.comm = comm
        # dictionary = comm.vb_id_vocab
        # dictionary = comm.gpt2_hf_tok
        # num_embeddings = len(dictionary)
        # padding_idx = dictionary.pad_token_id
        args = cfg.tx_dec
        # embed_dim = args.encoder_embed_dim
        # embed_toks = nn.Embedding(num_embeddings, embed_dim, padding_idx)

        super().__init__(
            d_model=1024,
            n_vocab_src=0,
            vocab_trg=0,
            d_hidden=1024,
            n_layers=args.encoder_layers,
            n_heads=args.encoder_attention_heads,
            drop_ratio=args.dropout,
            pe=False,
        )

    def forward(
        self,
        src_tokens=None,
        src_lengths=None,
        return_all_hiddens=False,
        token_embeddings=None,
    ):
        assert token_embeddings is not None
        enc_out = self.encoder(token_embeddings)[-1]

        return EncoderOut(
            encoder_out=enc_out.transpose(0, 1).contiguous(),
            encoder_padding_mask=None,
            encoder_embedding=None,
            encoder_states=None,
            src_tokens=None,
            src_lengths=None,
        )


def get_enc_out_base(enc_out):
    return EncoderOut(
        encoder_out=enc_out,  # T x B x C
        encoder_padding_mask=None,  # B x T
        encoder_embedding=None,  # B x T x C
        encoder_states=None,  # List[T x B x C]
        src_tokens=None,
        src_lengths=None,
    )


class TxEncoderNew_Conc(TxEncoderOld):
    def after_init(self):
        self.orig_tx_out_comb = nn.Sequential(
            *[nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, 1024)]
        )
        return

    def forward(
        self,
        src_tokens,
        src_lengths,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        tx_out = super().forward(
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
            token_embeddings=token_embeddings,
        )
        # B x T x C
        enc_out = tx_out.encoder_out.transpose(0, 1).contiguous()
        enc_out2 = torch.cat([token_embeddings, enc_out], dim=-1)

        enc_out3 = self.orig_tx_out_comb(enc_out2)
        return get_enc_out_base(enc_out=enc_out3.transpose(0, 1).contiguous())


def TxEncoder(cfg, comm):
    if cfg.mdl.tx_enc_type == "old":
        return TxEncoderOld(cfg, comm)
    elif cfg.mdl.tx_enc_type == "new":
        return TxEncoderNew(cfg, comm)
    elif cfg.mdl.tx_enc_type == "new_conc":
        return TxEncoderNew_Conc(cfg, comm)

    else:
        raise NotImplementedError


class TxDecoderReal(TransformerDecoder):
    def __init__(self, cfg, comm):
        self.full_cfg = cfg
        self.comm = comm
        dictionary = comm.gpt2_hf_tok
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad_token_id
        args = cfg.tx_dec
        embed_dim = args.decoder_embed_dim
        embed_toks = nn.Embedding(num_embeddings, embed_dim, padding_idx)

        super().__init__(args, dictionary, embed_toks)


class GPT2_hf_fseqDec(HuggingFaceGPT2Decoder):
    def __init__(self, cfg, comm):
        self.full_cfg = cfg
        self.comm = comm
        dictionary = comm.gpt2_hf_tok
        args = cfg
        super().__init__(args, dictionary)


def TxDecoder(full_cfg, comm):
    if full_cfg.mdl.tx_dec_type == "gpt2":
        return GPT2_hf_fseqDec(full_cfg, comm)
    elif full_cfg.mdl.tx_dec_type == "txdec":
        return TxDecoderReal(full_cfg, comm)
    else:
        raise NotImplementedError


class Simple_GPT2(nn.Module):
    """
    Simply Run a GPT2 model
    Assumes Verbs are given
    """

    def __init__(self, cfg, comm):
        super().__init__()
        self.full_cfg = cfg
        self.cfg = cfg.mdl
        self.comm = comm
        self.build_model()

    def build_model(self):
        self.gpt2_mdl = GPT2LMHeadModel.from_pretrained(self.cfg.gpt2_mdl_name)
        self.voc_size = len(self.comm.gpt2_hf_tok)
        self.gpt2_mdl.resize_token_embeddings(self.voc_size)
        self.pad_index = self.comm.gpt2_hf_tok.pad_token_id
        self.bos_index = self.comm.gpt2_hf_tok.eos_token_id
        return

    def forward_gen(self, inp, *args):
        src_toks1 = inp["seq_out_by_ev"][:, :, [0], :]
        B, num_ev, num_seq_eg, seq_len = src_toks1.shape
        src_toks = src_toks1.view(B * num_ev, num_seq_eg * seq_len)
        inp_ids = src_toks[..., :1].contiguous()

        wvoc = self.comm.gpt2_hf_tok
        out_sents = self.gpt2_mdl.generate(
            input_ids=inp_ids,
            max_length=60,
            use_cache=True,
            num_beams=1,
            num_return_sequences=1,
            do_sample=False,
            pad_token_id=wvoc.pad_token_id,
        )
        out_sents = out_sents.view(B, num_ev, num_seq_eg, -1)
        return out_sents

    def forward(self, inp):
        src_toks1 = inp["seq_out_by_ev"][:, :, [0], :]
        src_attn1 = inp["seq_out_lens_by_ev"][:, :, [0], :]
        B, num_ev, num_seq_eg, seq_len = src_toks1.shape
        assert num_seq_eg == 1
        src_toks = src_toks1.view(B * num_ev, num_seq_eg * seq_len)
        src_attn_mask = src_attn1.view(B * num_ev, num_seq_eg * seq_len)
        out = self.gpt2_mdl(
            input_ids=src_toks, attention_mask=src_attn_mask, return_dict=True,
        )
        # B*num_ev x num_seq_eg*seq_len x vocab_size
        logits = out["logits"]

        # out contains logits, past_key_vals
        # logits of shape: B x seq_len x vocab_size

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = src_toks[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.pad_index,
        )
        out["loss"] = loss
        return out


class GPT2_New(GPT2LMHeadModel):
    def prepare_inputs_for_generation(
        self, input_ids, past=None, attention_mask=None, **kwargs
    ):
        # only last token for inputs_ids if past is defined in kwargs
        if past is None:
            if "vid_emb" in kwargs:
                vid_emb = kwargs.pop("vid_emb")
                input_embs = self.transformer.wte(input_ids)
                input_embs_new = torch.cat([vid_emb, input_embs], dim=1)
                return {
                    "inputs_embeds": input_embs_new,
                    "past_key_values": past,
                    "use_cache": kwargs.get("use_cache"),
                }
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
        }


class Simple_GPT2_New(Simple_GPT2):
    def build_model(self):
        self.gpt2_mdl = GPT2_New.from_pretrained(self.cfg.gpt2_mdl_name)
        self.voc_size = len(self.comm.gpt2_hf_tok)
        self.gpt2_mdl.resize_token_embeddings(self.voc_size)
        self.pad_index = self.comm.gpt2_hf_tok.pad_token_id
        self.bos_index = self.comm.gpt2_hf_tok.eos_token_id
        return

    def forward_gen(self, inp, *args):
        src_toks1 = inp["seq_out_by_ev"][:, :, [0], :]
        B, num_ev, num_seq_eg, seq_len = src_toks1.shape
        src_toks = src_toks1.view(B * num_ev, num_seq_eg * seq_len)
        inp_ids = src_toks[..., :1].contiguous()

        wvoc = self.comm.gpt2_hf_tok

        out_sents = self.gpt2_mdl.generate(
            input_ids=inp_ids,
            max_length=60 + inp_ids.size(-1),
            use_cache=True,
            num_beams=1,
            num_return_sequences=1,
            do_sample=False,
            pad_token_id=wvoc.pad_token_id,
        )
        out_sents = out_sents.view(B, num_ev, num_seq_eg, -1)
        return out_sents


class Simple_TxDec(nn.Module):
    def __init__(self, cfg, comm):
        super(Simple_TxDec, self).__init__()
        self.full_cfg = cfg
        self.cfg = cfg.mdl
        self.sf_cfg = cfg.sf_mdl

        self.comm = comm
        self.use_encoder = False
        self.build_model()

    def build_model(self):
        self.decoder = TxDecoder(self.full_cfg, self.comm)
        self.pad_index = self.comm.gpt2_hf_tok.pad_token_id
        self.bos_index = self.comm.gpt2_hf_tok.eos_token_id
        self.max_decoder_positions = lambda: 1024
        self.get_normalized_probs = self.decoder.get_normalized_probs
        return

    def forward_encoder(self, inp):
        return None

    def prepare_prev_toks_inp(self, inp):
        dst_toks1 = inp["seq_out_by_ev"][:, :, [0], :]
        dst_attn1 = inp["seq_out_lens_by_ev"][:, :, [0], :]
        vb_toks1 = inp["vb_out_by_ev"][:, :, [0], :]

        B, num_ev, num_seq_eg, seq_len = dst_toks1.shape
        assert num_seq_eg == 1
        dst_toks = dst_toks1.view(B * num_ev, num_seq_eg * seq_len)
        dst_attn_mask = dst_attn1.view(B * num_ev, num_seq_eg * seq_len)
        dst_lens = dst_attn_mask.sum(dim=-1)

        vb_toks = vb_toks1.view(B * num_ev, num_seq_eg * vb_toks1.size(-1))
        return {"dst_toks": dst_toks, "dst_lens": dst_lens, "vb_only_tokens": vb_toks}

    def forward_decoder(
        self, prev_tokens, encoder_out, incremental_state=None, temperature=None
    ):
        if isinstance(encoder_out, list) and len(encoder_out) == 0:
            encoder_out = None
        decoder_out = self.decoder(
            prev_tokens, encoder_out=encoder_out, incremental_state=incremental_state
        )
        return decoder_out

    def forward(self, inp):
        inp_prep = self.prepare_prev_toks_inp(inp)
        encoder_out = self.forward_encoder(inp)
        prev_tokens = inp_prep["dst_toks"]

        decoder_out = self.forward_decoder(
            prev_tokens=prev_tokens, encoder_out=encoder_out
        )
        logits = decoder_out[0]
        shift_logits = logits[..., :-1, :].contiguous()
        labels = inp_prep["dst_toks"]
        shifted_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, logits.size(-1)),
            shifted_labels.view(-1),
            ignore_index=self.pad_index,
        )
        out_dct = {"loss": loss, "logits": logits}

        return out_dct

    def forward_gen(self, inp, seq_gen: SeqGenCustom):
        inp_prep = self.prepare_prev_toks_inp(inp)

        inp["src_tokens"] = inp_prep["dst_toks"][..., :1]
        inp["src_lengths"] = inp_prep["dst_lens"]
        inp_ids = inp_prep["dst_toks"][..., :1]
        out_sents = seq_gen._generate(inp, prefix_tokens=inp_ids)
        src_toks1 = inp["seq_out_by_ev"][:, :, [0], :]
        B, num_ev, num_seq_eg, seq_len = src_toks1.shape

        max_len = max([len(o[0]["tokens"]) for o in out_sents])
        B1 = inp_ids.size(0)
        out_sents_tensor = inp_ids.new_full((B1, max_len), self.pad_index)
        for ix in range(B1):
            xtoks = out_sents[ix][0]["tokens"]
            out_sents_tensor[ix, : len(xtoks)] = xtoks

        out_sents1 = out_sents_tensor.view(B, num_ev, num_seq_eg, -1)
        return out_sents1


class Simple_TxEncDec(Simple_TxDec):
    def build_model(self):
        super().build_model()
        self.encoder = TxEncoder(self.full_cfg, self.comm)
        self.use_encoder = True
        return

    def forward_encoder(self, inp):
        src_toks = inp["src_tokens"]
        src_lens = inp["src_lengths"]
        encoder_out = self.encoder(
            src_toks, src_lengths=src_lens, return_all_hiddens=True
        )
        return encoder_out


class Reorderer:
    def reorder_encoder_out(self, encoder_out: EncoderOut, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        """
        Since encoder_padding_mask and encoder_embedding are both of type
        Optional[Tensor] in EncoderOut, they need to be copied as local
        variables for Torchscript Optional refinement
        """
        encoder_padding_mask = encoder_out.encoder_padding_mask
        encoder_embedding = encoder_out.encoder_embedding
        new_encoder_out = (
            encoder_out.encoder_out
            if encoder_out.encoder_out is None
            else encoder_out.encoder_out.index_select(1, new_order)
        )
        new_encoder_padding_mask = (
            encoder_padding_mask
            if encoder_padding_mask is None
            else encoder_padding_mask.index_select(0, new_order)
        )
        new_encoder_embedding = (
            encoder_embedding
            if encoder_embedding is None
            else encoder_embedding.index_select(0, new_order)
        )
        src_tokens = encoder_out.src_tokens
        if src_tokens is not None:
            src_tokens = src_tokens.index_select(0, new_order)

        src_lengths = encoder_out.src_lengths
        if src_lengths is not None:
            src_lengths = src_lengths.index_select(0, new_order)

        encoder_states = encoder_out.encoder_states
        if encoder_states is not None:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return EncoderOut(
            encoder_out=new_encoder_out,  # T x B x C
            encoder_padding_mask=new_encoder_padding_mask,  # B x T
            encoder_embedding=new_encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=src_tokens,  # B x T
            src_lengths=src_lengths,  # B x 1
        )


def get_head_dim(full_cfg) -> int:
    if "i3d" in full_cfg.ds.vsitu.vsit_frm_feats_dir:
        head_dim = 2048
    elif ("slow_fast" in full_cfg.ds.vsitu.vsit_frm_feats_dir) or (
        "sfast" in full_cfg.ds.vsitu.vsit_frm_feats_dir
    ):
        head_dim = 2304
    else:
        raise NotImplementedError
    return head_dim


class SFPreFeats_TxDec(Simple_TxDec, Reorderer):
    def build_model(self):
        super().build_model()
        head_dim = get_head_dim(self.full_cfg)
        self.vid_feat_encoder = nn.Sequential(
            *[nn.Linear(head_dim, 1024), nn.ReLU(), nn.Linear(1024, 1024)]
        )
        self.use_encoder = True

        return

    def forward_encoder(self, inp):
        frm_feats = inp["frm_feats"]
        B = inp["vseg_idx"].size(0)
        assert frm_feats.size(1) == 5
        out = self.vid_feat_encoder(frm_feats)
        out = out.view(B * 5, 1, -1)

        encoder_out = EncoderOut(
            encoder_out=out.transpose(0, 1).contiguous(),  # 5 x B x vdim,
            encoder_padding_mask=None,
            encoder_embedding=None,
            encoder_states=None,
            src_tokens=None,
            src_lengths=None,
        )

        return encoder_out


class SFPreFeats_TxEncDec(Simple_TxDec, Reorderer):
    def build_model(self):
        super().build_model()
        head_dim = get_head_dim(self.full_cfg)

        self.vid_feat_encoder = nn.Sequential(
            *[nn.Linear(head_dim, 1024), nn.ReLU(), nn.Linear(1024, 1024)]
        )
        self.use_encoder = True

        self.vid_feat_txenc = TxEncoder(self.full_cfg, self.comm)
        return

    def forward_encoder(self, inp):
        frm_feats = inp["frm_feats"]
        B = inp["vseg_idx"].size(0)
        assert frm_feats.size(1) == 5
        out = self.vid_feat_encoder(frm_feats)
        out = out.view(B, 5, -1)

        tx_out = self.vid_feat_txenc(
            src_tokens=out[..., 0],
            src_lengths=None,
            return_all_hiddens=True,
            token_embeddings=out,
        )
        enc_out_batch1 = tx_out.encoder_out.transpose(0, 1).contiguous()
        enc_out2 = enc_out_batch1.view(B * 5, 1, -1)
        enc_out3 = enc_out2.transpose(0, 1).contiguous()

        encoder_out = EncoderOut(
            encoder_out=enc_out3,  # 1 x 5*B x vdim,
            encoder_padding_mask=None,
            encoder_embedding=None,
            encoder_states=None,
            src_tokens=None,
            src_lengths=None,
        )

        return encoder_out
