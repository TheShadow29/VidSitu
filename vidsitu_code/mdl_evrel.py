"""
Model for EvRel
"""
import torch
from torch import nn
from torch.nn import functional as F

from vidsitu_code.mdl_sf_base import get_head_dim
from transformers import RobertaForSequenceClassification, RobertaModel


class Simple_EvRel_Roberta(nn.Module):
    def __init__(self, cfg, comm):
        super().__init__()
        self.full_cfg = cfg
        self.cfg = cfg.mdl
        self.comm = comm
        self.build_model()

    def build_model(self):
        self.rob_mdl = RobertaForSequenceClassification.from_pretrained(
            self.full_cfg.mdl.rob_mdl_name, num_labels=5
        )
        return

    def forward(self, inp):
        src_toks1 = inp["evrel_seq_out"]
        src_attn1 = inp["evrel_seq_out_lens"]

        B, num_ev, num_seq_eg, seq_len = src_toks1.shape
        src_toks = src_toks1.view(B * num_ev * num_seq_eg, seq_len)
        src_attn_mask = src_attn1.view(B * num_ev * num_seq_eg, seq_len)

        out = self.rob_mdl(
            input_ids=src_toks,
            attention_mask=src_attn_mask,
            return_dict=True,
            # token_type_ids=src_tok_typ_ids,
        )
        # B*num_ev x num_seq_eg*seq_len x vocab_size
        logits = out["logits"]
        labels = inp["evrel_labs"]
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            # ignore_index=self.pad_index,
        )
        out["loss"] = loss
        out["mdl_out"] = logits.view(B, num_ev, num_seq_eg, -1)
        return out


class SFPret_SimpleEvRel(nn.Module):
    def __init__(self, cfg, comm):
        super().__init__()
        self.full_cfg = cfg
        self.cfg = cfg.mdl
        self.comm = comm
        self.build_model()

    def build_model(self):
        self.rob_mdl = RobertaModel.from_pretrained(
            self.full_cfg.mdl.rob_mdl_name, add_pooling_layer=True
        )
        head_dim = get_head_dim(self.full_cfg)

        self.vid_feat_encoder = nn.Sequential(
            *[nn.Linear(head_dim, 1024), nn.ReLU(), nn.Linear(1024, 1024)]
        )

        self.vis_lang_encoder = nn.Sequential(
            *[nn.Linear(1792, 1024), nn.ReLU(), nn.Linear(1024, 1024)]
        )
        self.vis_lang_classf = nn.Sequential(
            *[nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, 5)]
        )
        return

    def get_src(self, inp):
        return inp["evrel_seq_out_ones"], inp["evrel_seq_out_ones_lens"]

    def forward(self, inp):
        src_toks1, src_attn1 = self.get_src(inp)

        B, num_ev, num_seq_eg, seq_len = src_toks1.shape
        src_toks = src_toks1.view(B * num_ev * num_seq_eg, seq_len)
        src_attn_mask = src_attn1.view(B * num_ev * num_seq_eg, seq_len)

        lang_out = self.rob_mdl(
            input_ids=src_toks, attention_mask=src_attn_mask, return_dict=True
        )
        pooler_out = lang_out.pooler_output

        pooler_out_5 = pooler_out.view(B, 5, num_seq_eg, pooler_out.size(-1))

        frm_feats = inp["frm_feats"]
        B = inp["vseg_idx"].size(0)
        assert frm_feats.size(1) == 5
        vis_out = self.vid_feat_encoder(frm_feats)
        vis_out = (
            vis_out.view(B, 5, 1, -1)
            .contiguous()
            .expand(B, 5, num_seq_eg, -1)
            .contiguous()
        )
        # vis_lang_out = self.vis_lang_encoder(
        #     torch.cat([vis_out.new_zeros(vis_out.shape), pooler_out_5], dim=-1)
        # )
        vis_lang_out = self.vis_lang_encoder(torch.cat([vis_out, pooler_out_5], dim=-1))
        vis_lang_out1 = torch.index_select(
            vis_lang_out, dim=1, index=vis_lang_out.new_tensor([0, 1, 2, 2]).long()
        ).contiguous()
        vis_lang_out2 = torch.index_select(
            vis_lang_out, dim=1, index=vis_lang_out.new_tensor([2, 2, 3, 4]).long()
        ).contiguous()

        vis_lang_out3 = torch.cat([vis_lang_out1, vis_lang_out2], dim=-1)
        logits = self.vis_lang_classf(vis_lang_out3).contiguous()

        # B*num_ev x num_seq_eg*seq_len x vocab_size
        labels = inp["evrel_labs"]
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            # ignore_index=self.pad_index,
        )
        out_dct = {}
        out_dct["loss"] = loss
        out_dct["mdl_out"] = logits.view(B, num_ev - 1, num_seq_eg, -1)
        return out_dct


class SFPret_OnlyVb_SimpleEvRel(SFPret_SimpleEvRel):
    def get_src(self, inp):
        return inp["evrel_vbonly_out_ones"], inp["evrel_vbonly_out_ones_lens"]


class SFPret_OnlyVid_SimpleEvRel(SFPret_SimpleEvRel):
    def forward(self, inp):
        src_toks1, src_attn1 = self.get_src(inp)
        # src_tok_typ_ids1 = inp["evrel_seq_tok_ids"]
        B, num_ev, num_seq_eg, seq_len = src_toks1.shape
        src_toks = src_toks1.view(B * num_ev * num_seq_eg, seq_len)
        src_attn_mask = src_attn1.view(B * num_ev * num_seq_eg, seq_len)
        # src_tok_typ_ids = src_tok_typ_ids1.view(B * num_ev * num_seq_eg, seq_len)
        lang_out = self.rob_mdl(
            input_ids=src_toks, attention_mask=src_attn_mask, return_dict=True
        )
        pooler_out = lang_out.pooler_output

        pooler_out_5 = pooler_out.view(B, 5, num_seq_eg, pooler_out.size(-1))

        frm_feats = inp["frm_feats"]
        B = inp["vseg_idx"].size(0)
        assert frm_feats.size(1) == 5
        vis_out = self.vid_feat_encoder(frm_feats)
        vis_out = (
            vis_out.view(B, 5, 1, -1)
            .contiguous()
            .expand(B, 5, num_seq_eg, -1)
            .contiguous()
        )

        vis_lang_out = self.vis_lang_encoder(
            torch.cat([vis_out, pooler_out_5.new_zeros(pooler_out_5.shape)], dim=-1)
        )
        vis_lang_out1 = torch.index_select(
            vis_lang_out, dim=1, index=vis_lang_out.new_tensor([0, 1, 2, 2]).long()
        ).contiguous()
        vis_lang_out2 = torch.index_select(
            vis_lang_out, dim=1, index=vis_lang_out.new_tensor([2, 2, 3, 4]).long()
        ).contiguous()

        vis_lang_out3 = torch.cat([vis_lang_out1, vis_lang_out2], dim=-1)
        logits = self.vis_lang_classf(vis_lang_out3).contiguous()

        # B*num_ev x num_seq_eg*seq_len x vocab_size

        labels = inp["evrel_labs"]
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            # ignore_index=self.pad_index,
        )
        out_dct = {}
        out_dct["loss"] = loss
        out_dct["mdl_out"] = logits.view(B, num_ev - 1, num_seq_eg, -1)
        return out_dct


class Simple_TxEncEvRel(SFPret_SimpleEvRel):
    def forward(self, inp):
        src_toks1, src_attn1 = self.get_src(inp)
        B, num_ev, num_seq_eg, seq_len = src_toks1.shape
        src_toks = src_toks1.view(B * num_ev * num_seq_eg, seq_len)
        src_attn_mask = src_attn1.view(B * num_ev * num_seq_eg, seq_len)

        lang_out = self.rob_mdl(
            input_ids=src_toks, attention_mask=src_attn_mask, return_dict=True
        )
        pooler_out = lang_out.pooler_output

        pooler_out_5 = pooler_out.view(B, 5, num_seq_eg, pooler_out.size(-1))

        frm_feats = inp["frm_feats"]
        B = inp["vseg_idx"].size(0)
        assert frm_feats.size(1) == 5
        vis_out = self.vid_feat_encoder(frm_feats)
        vis_out = (
            vis_out.view(B, 5, 1, -1)
            .contiguous()
            .expand(B, 5, num_seq_eg, -1)
            .contiguous()
        )

        vis_lang_out = self.vis_lang_encoder(
            torch.cat([vis_out.new_zeros(vis_out.shape), pooler_out_5], dim=-1)
        )
        vis_lang_out1 = torch.index_select(
            vis_lang_out, dim=1, index=vis_lang_out.new_tensor([0, 1, 2, 2]).long()
        ).contiguous()
        vis_lang_out2 = torch.index_select(
            vis_lang_out, dim=1, index=vis_lang_out.new_tensor([2, 2, 3, 4]).long()
        ).contiguous()

        vis_lang_out3 = torch.cat([vis_lang_out1, vis_lang_out2], dim=-1)
        logits = self.vis_lang_classf(vis_lang_out3).contiguous()

        # B*num_ev x num_seq_eg*seq_len x vocab_size

        labels = inp["evrel_labs"]
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            # ignore_index=self.pad_index,
        )
        out_dct = {}
        out_dct["loss"] = loss
        out_dct["mdl_out"] = logits.view(B, num_ev - 1, num_seq_eg, -1)
        return out_dct
