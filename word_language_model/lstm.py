from typing import Any

import torch
import torch.nn as nn
from typing import Tuple

"""
Implementation of LSTM model described in On the State of the Art of Evaluation in Neural Language Models
https://arxiv.org/abs/1707.05589
"""


class LSTM(nn.Module):
  """
  The LSTM expects the input to be batch first: Batch size x length
  The LSTM does not support bidirectional as an argument
  """
  __constants__ = ['layers', 'hidden_size', 'skip_connection']

  def __init__(self, input_size, hidden_size,
               num_layers=1, bias=True,
               inter_layer_dropout=0., recurrent_dropout=0.,
               skip_connection=False):
    super(LSTM, self).__init__()

    assert num_layers >= 1 and inter_layer_dropout >= 0. and recurrent_dropout >= 0.

    self.layers = nn.ModuleList([
      nn.LSTMCell(input_size=input_size, hidden_size=hidden_size, bias=bias)
    ])
    for layer in range(num_layers - 1):
      self.layers.append(nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size, bias=bias))

    self.inter_layer_dropout = nn.Dropout(p=inter_layer_dropout)
    self.recurrent_dropout = nn.Dropout(p=recurrent_dropout)

    self.num_layers = num_layers
    self.hidden_size = hidden_size
    self.skip_connection = skip_connection

  def forward(self, input: torch.Tensor, hx: Tuple[torch.Tensor, torch.Tensor] = None):
    """

    :param input: batch size x length x input_size
    :param hx: num_layers x batch_size x hidden_size
    :return: batch size x length x input_size
    """
    input_length = input.shape[1]
    batch_size = input.shape[0]
    # time x num_layers x batch_size x hidden_state
    lstm_h_list, lstm_c_list = input.new_full(size=(input_length + 1, self.num_layers,
                                                    batch_size, self.hidden_size), fill_value=0.), \
                               input.new_full(size=(input_length + 1, self.num_layers,
                                                    batch_size, self.hidden_size), fill_value=0.)

    lstm_h_list[0] = hx[0]
    lstm_c_list[0] = hx[1]

    layer_index = -1
    for layer in self.layers:
      layer_index += 1
      for index in range(input.shape[1]):
        if layer_index == 0:
          lstm_input = input.select(dim=1, index=index)
        else:
          lstm_input = lstm_h_list[index + 1][layer_index - 1]
          lstm_input = self.inter_layer_dropout(lstm_input)
        h_0, c_0 = lstm_h_list[index][layer_index], lstm_c_list[index][layer_index]
        h_0, c_0 = self.recurrent_dropout(h_0), self.recurrent_dropout(c_0)
        h_1, c_1 = layer(lstm_input, (h_0, c_0))
        lstm_h_list[index + 1][layer_index] = h_1
        lstm_c_list[index + 1][layer_index] = c_1

    output = lstm_h_list[1:]

    if self.skip_connection:
      output = output.sum(dim=1)
    else:
      output = output[:, -1, :, :]

    output = output.permute(1, 0, 2)
    return output, (lstm_h_list[-1], lstm_c_list[-1])


class RNNLM(nn.Module):
  """
  RNN LM input dropout, intra-layer dropout, output dropout, recurrent states
  # rnn_type, ntoken, ninp, nhid, nlayers, dropout = 0.5, tie_weights = False
  # input_size, hidden_size,
  # num_layers = 1, bias = True,
  # inter_layer_dropout = 0., recurrent_dropout = 0.
  """

  def __init__(self, num_tokens, input_size, hidden_size,
               num_layers=1, bias=True,
               inter_layer_dropout=0., recurrent_dropout=0.,
               input_dropout=0., output_dropout=0.,
               tie_weights=False):
    super(RNNLM, self).__init__()

    self.embedding = nn.Embedding(num_embeddings=num_tokens, embedding_dim=input_size)

    self.lstm = LSTM(input_size=input_size, hidden_size=hidden_size,
                     num_layers=num_layers, bias=bias,
                     inter_layer_dropout=inter_layer_dropout, recurrent_dropout=recurrent_dropout)

    self.input_dropout = nn.Dropout(p=input_dropout)
    self.output_dropout = nn.Dropout(p=output_dropout)

    self.decoder = nn.Linear(hidden_size, num_tokens)

    if tie_weights:
      if input_size != hidden_size:
        raise ValueError('When using the tied flag, nhid must be equal to emsize')
      self.decoder.weight = self.embedding.weight

    self.num_layers = num_layers
    self.hidden_size = hidden_size

  def init_weights(self):
    initrange = 0.1
    self.embedding.weight.data.uniform_(-initrange, initrange)
    self.decoder.bias.data.zero_()
    self.decoder.weight.data.uniform_(-initrange, initrange)

  def forward(self, input, hidden):
    """

    :param input: batch size x length
    :param hidden: num_layers x batch_size x hidden_size
    :return:
    """
    embedded = self.input_dropout(self.embedding(input))
    # making the input batch first
    embedded = embedded.permute(1, 0, 2)
    output, hidden = self.lstm(embedded, hidden)
    output = self.output_dropout(output)
    decoded = self.decoder(output)
    return decoded, hidden

  def init_hidden(self, bsz):
    weight = next(self.parameters())
    return (weight.new_zeros(self.num_layers, bsz, self.hidden_size),
            weight.new_zeros(self.num_layers, bsz, self.hidden_size))
