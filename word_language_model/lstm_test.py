import unittest
import torch
import torch.nn
import lstm
import torch.testing as torch_testing
import random
import numpy as np
import torch.jit


class CustomLSTMTest(unittest.TestCase):

  def test_lstm_basic_single_layer(self):
    self.set_seed()
    # :param input: batch size x length x input_size
    # :param hx: num_layers x batch_size x hidden_size

    batch_size = 13
    length = 4
    input_size = 32
    hidden_size = 64
    num_layers = 1
    bias = True
    inter_layer_dropout = 0.
    recurrent_dropout = 0.
    input = torch.rand(batch_size, length, input_size)
    hx = torch.rand(num_layers, batch_size, hidden_size), torch.rand(num_layers, batch_size, hidden_size)

    torch_lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                               num_layers=num_layers, bias=bias, dropout=inter_layer_dropout,
                               batch_first=True, bidirectional=False)

    custom_lstm = lstm.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, bias=bias,
                            inter_layer_dropout=inter_layer_dropout, recurrent_dropout=recurrent_dropout)

    self.copy_weights_from_torch_to_custom(torch_lstm, custom_lstm)

    torch_output, torch_hidden = torch_lstm(input, hx)
    custom_output, custom_hidden = custom_lstm(input, hx)

    torch_testing.assert_allclose(torch_hidden[0], custom_hidden[0])
    torch_testing.assert_allclose(torch_hidden[1], custom_hidden[1])

    torch_testing.assert_allclose(custom_output, torch_output)

  def test_lstm_basic_multi_layer(self):
    self.set_seed()
    # :param input: batch size x length x input_size
    # :param hx: num_layers x batch_size x hidden_size

    batch_size = 13
    length = 7
    input_size = 32
    hidden_size = 64
    num_layers = 4
    bias = True
    inter_layer_dropout = 0.
    recurrent_dropout = 0.
    skip_connection = False

    input = torch.rand(batch_size, length, input_size)
    hx = torch.rand(num_layers, batch_size, hidden_size), torch.rand(num_layers, batch_size, hidden_size)

    torch_lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                               num_layers=num_layers, bias=bias, dropout=inter_layer_dropout,
                               batch_first=True, bidirectional=False)

    custom_lstm = lstm.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, bias=bias,
                            inter_layer_dropout=inter_layer_dropout, recurrent_dropout=recurrent_dropout,
                            skip_connection=skip_connection)

    self.copy_weights_from_torch_to_custom(torch_lstm, custom_lstm)

    torch_output, torch_hidden = torch_lstm(input, hx)
    custom_output, custom_hidden = custom_lstm(input, hx)

    torch_testing.assert_allclose(torch_hidden[0], custom_hidden[0])
    torch_testing.assert_allclose(torch_hidden[1], custom_hidden[1])

    torch_testing.assert_allclose(custom_output, torch_output)

  def test_lstm_basic_multi_layer_without_bias(self):
    self.set_seed()
    # :param input: batch size x length x input_size
    # :param hx: num_layers x batch_size x hidden_size

    batch_size = 13
    length = 7
    input_size = 32
    hidden_size = 64
    num_layers = 4
    bias = False
    inter_layer_dropout = 0.
    recurrent_dropout = 0.
    skip_connection = False

    input = torch.rand(batch_size, length, input_size)
    hx = torch.rand(num_layers, batch_size, hidden_size), torch.rand(num_layers, batch_size, hidden_size)

    torch_lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                               num_layers=num_layers, bias=bias, dropout=inter_layer_dropout,
                               batch_first=True, bidirectional=False)

    custom_lstm = lstm.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, bias=bias,
                            inter_layer_dropout=inter_layer_dropout, recurrent_dropout=recurrent_dropout,
                            skip_connection=skip_connection)

    self.copy_weights_from_torch_to_custom(torch_lstm, custom_lstm, bias=bias)

    torch_output, torch_hidden = torch_lstm(input, hx)
    custom_output, custom_hidden = custom_lstm(input, hx)

    torch_testing.assert_allclose(torch_hidden[0], custom_hidden[0])
    torch_testing.assert_allclose(torch_hidden[1], custom_hidden[1])

    torch_testing.assert_allclose(custom_output, torch_output)

  def test_lstm_basic_single_layer_jit(self):
    self.set_seed()
    # :param input: batch size x length x input_size
    # :param hx: num_layers x batch_size x hidden_size

    batch_size = 13
    length = 4
    input_size = 32
    hidden_size = 64
    num_layers = 1
    bias = True
    inter_layer_dropout = 0.
    recurrent_dropout = 0.
    input = torch.rand(batch_size, length, input_size)
    hx = torch.rand(num_layers, batch_size, hidden_size), torch.rand(num_layers, batch_size, hidden_size)

    torch_lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                               num_layers=num_layers, bias=bias, dropout=inter_layer_dropout,
                               batch_first=True, bidirectional=False)

    custom_lstm = lstm.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, bias=bias,
                            inter_layer_dropout=inter_layer_dropout, recurrent_dropout=recurrent_dropout)

    self.copy_weights_from_torch_to_custom(torch_lstm, custom_lstm)

    custom_lstm = torch.jit.script(custom_lstm)

    torch_output, torch_hidden = torch_lstm(input, hx)
    custom_output, custom_hidden = custom_lstm(input, hx)

    torch_testing.assert_allclose(torch_hidden[0], custom_hidden[0])
    torch_testing.assert_allclose(torch_hidden[1], custom_hidden[1])

    torch_testing.assert_allclose(custom_output, torch_output)

  @staticmethod
  def copy_weights_from_torch_to_custom(torch_model, custom_model, bias=True):
    c = 0
    parameter_list = list(torch_model.parameters())
    for i in range(len(custom_model.layers)):
      custom_model.layers[i].weight_ih = parameter_list[c]
      custom_model.layers[i].weight_hh = parameter_list[c + 1]
      if bias:
        custom_model.layers[i].bias_ih = parameter_list[c + 2]
        custom_model.layers[i].bias_hh = parameter_list[c + 3]
      c += 4 if bias else 2

  @staticmethod
  def set_seed(seed=0):
    torch.random.manual_seed(seed)
    # random.seed(seed)
    # np.random.seed(seed)


if __name__ == '__main__':
  unittest.main()
