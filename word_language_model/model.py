import logging
import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import lstm


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, num_tokens,
                 embedding_size, hidden_size,
                 num_layers,
                 inter_layer_dropout=0., recurrent_dropout=0.,
                 input_dropout=0., output_dropout=0.,
                 up_project_embedding=False,
                 up_project_hidden=False,
                 tie_weights=False,
                 lstm_skip_connection=False):
        super(RNNModel, self).__init__()

        self.input_dropout = nn.Dropout(input_dropout)

        encoder_layer_list = []
        embedding_layer = nn.Embedding(num_tokens, embedding_size)

        encoder_layer_list.append(embedding_layer)
        if embedding_size != hidden_size and up_project_embedding:
            logging.info("Encoder: adding linear transformation to up project embedding to hidden")
            encoder_layer_list.append(nn.Linear(embedding_size, hidden_size, bias=False))
            rnn_input_size = hidden_size
        else:
            rnn_input_size = embedding_size
        self.encoder = nn.Sequential(*encoder_layer_list)

        if rnn_type in ['LSTM', 'GRU']:
            if recurrent_dropout > 0.:
                logging.warning("recurrent_dropout argument is only used in the custom LSTM model")

            self.rnn = getattr(nn, rnn_type)(rnn_input_size, hidden_size, num_layers, dropout=inter_layer_dropout)
        elif rnn_type == "custom_LSTM":
            self.rnn = lstm.LSTM(rnn_input_size, hidden_size,
                                 num_layers, bias=True,
                                 inter_layer_dropout=inter_layer_dropout, recurrent_dropout=recurrent_dropout,
                                 skip_connection=lstm_skip_connection, batch_first=False)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(rnn_input_size, hidden_size, num_layers, nonlinearity=nonlinearity,
                              dropout=inter_layer_dropout)
        self.output_dropout = nn.Dropout(output_dropout)

        decoder_list = []
        linear_layer = nn.Linear(embedding_size, num_tokens, bias=False)
        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        self.decoder_is_sequential = True
        if tie_weights:
            linear_layer.weight = embedding_layer.weight

            if hidden_size != embedding_size:
                if up_project_hidden:
                    self.decoder_is_sequential = False
                    logging.info("Decoder: adding linear transformation to up project embedding to to hidden")
                    decoder_list.append(nn.Linear(embedding_size, hidden_size))
                else:
                    logging.info("Decoder: adding linear transformation to down project hidden to embedding")
                    decoder_list.append(nn.Linear(hidden_size, embedding_size))

        decoder_list.append(linear_layer)
        if self.decoder_is_sequential:
            self.decoder = nn.Sequential(*decoder_list)
        else:
            self.decoder = nn.ModuleList(decoder_list)

        self.rnn_type = rnn_type
        self.nhid = hidden_size
        self.nlayers = num_layers

        self.init_weights()

    def init_weights(self):
        initrange = 1 / self.nhid

        def init_layers(layer):
            if isinstance(layer, (torch.nn.Sequential, torch.nn.ModuleList)):
                [init_layers(l) for l in layer]
            else:
                layer.weight.data.uniform_(-initrange, initrange)
                if hasattr(layer, "bias") and layer.bias is not None:
                    layer.bias.data.zero_()

        init_layers(self.decoder)
        init_layers(self.encoder)

    def forward(self, input: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        emb = self.input_dropout(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.output_dropout(output)
        if self.decoder_is_sequential:
            decoded = self.decoder(output)
        else:
            embedding_weight = self.decoder[1].weight
            embedding_transform_layer = self.decoder[0]
            embedding_up_projected = embedding_transform_layer(embedding_weight)
            decoded = torch.matmul(output, embedding_up_projected.t())
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type in ['LSTM', "custom_LSTM"]:
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)


# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)
