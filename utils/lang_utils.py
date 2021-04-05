import torch
from torch import nn

# from fairseq.models import FairseqEncoder
from torch.nn import functional as F
from fairseq import utils


class LSTMEncoder(nn.Module):
    """LSTM encoder."""

    def __init__(
        self,
        cfg,
        comm,
        embed_dim=512,
        hidden_size=512,
        num_layers=1,
        dropout_in=0.1,
        dropout_out=0.1,
        bidirectional=False,
        left_pad=True,
        pretrained_embed=None,
        padding_value=0.0,
        num_embeddings=0,
        pad_idx=0,
    ):
        super().__init__()
        self.cfg = cfg
        self.comm = comm
        self.num_layers = num_layers
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        num_embeddings = num_embeddings
        self.padding_idx = pad_idx
        embed_dim1 = embed_dim
        self.embed_dim = embed_dim1
        if pretrained_embed is None:
            self.embed_tokens = nn.Embedding(
                num_embeddings, embed_dim1, self.padding_idx
            )
        else:
            self.embed_tokens = pretrained_embed

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.dropout_out if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.left_pad = left_pad
        self.padding_value = padding_value

        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2

    def forward(self, src_tokens=None, src_lengths=None, token_embeds=None):
        if self.left_pad:
            # nn.utils.rnn.pack_padded_sequence requires right-padding;
            # convert left-padding to right-padding
            src_tokens = utils.convert_padding_direction(
                src_tokens, self.padding_idx, left_to_right=True,
            )

        if token_embeds is None:
            bsz, seqlen = src_tokens.size()
            # embed tokens
            x = self.embed_tokens(src_tokens)
        else:

            x = token_embeds
            bsz, seqlen, embed_dim = token_embeds.shape
            assert embed_dim == self.embed_dim

        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # pack embedded source tokens into a PackedSequence

        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, src_lengths.cpu(), enforce_sorted=False
        )

        # apply LSTM
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        h0 = x.new_zeros(*state_size)
        c0 = x.new_zeros(*state_size)
        packed_outs, (final_hiddens, final_cells) = self.lstm(packed_x, (h0, c0))

        # unpack outputs and apply dropout
        x, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outs, padding_value=self.padding_value
        )
        x = F.dropout(x, p=self.dropout_out, training=self.training)
        assert list(x.size()) == [seqlen, bsz, self.output_units]

        if self.bidirectional:

            def combine_bidir(outs):
                out = (
                    outs.view(self.num_layers, 2, bsz, -1).transpose(1, 2).contiguous()
                )
                return out.view(self.num_layers, bsz, -1)

            final_hiddens = combine_bidir(final_hiddens)
            final_cells = combine_bidir(final_cells)

        if src_tokens is not None:
            encoder_padding_mask = src_tokens.eq(self.padding_idx).t()
        else:
            encoder_padding_mask = None

        return {
            "encoder_out": (x, final_hiddens, final_cells),
            "encoder_padding_mask": (
                encoder_padding_mask
                if encoder_padding_mask is not None and encoder_padding_mask.any()
                else None
            ),
        }

    def reorder_only_outputs(self, outputs):
        """
        outputs of shape : T x B x C -> B x T x C
        """
        return outputs.transpose(0, 1).contiguous()

    def reorder_encoder_out(self, encoder_out, new_order):
        encoder_out["encoder_out"] = tuple(
            eo.index_select(1, new_order) for eo in encoder_out["encoder_out"]
        )
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(1, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return int(1e5)  # an arbitrary large number
