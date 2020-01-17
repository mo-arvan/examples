import unittest

import torch
import torch.jit
import torch.nn
import torch.testing as torch_testing

import lstm


class CustomLSTMTest(unittest.TestCase):

    def test_lstm_basic_single_layer(self):
        self.set_seed()
        # :param input: batch size x length x input_size
        # :param hx: num_layers x batch_size x hidden_size

        batch_size = 13
        length = 2
        input_size = 32
        hidden_size = 64
        num_layers = 1
        bias = True
        inter_layer_dropout = 0.
        recurrent_dropout = 0.
        batch_first = True
        skip_connection = False
        jit_forward_custom = False

        self.compare_custom_and_cuda(batch_size, length, input_size, hidden_size, num_layers, bias,
                                     inter_layer_dropout, recurrent_dropout, batch_first, skip_connection,
                                     jit_forward_custom)

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
        batch_first = True
        jit_forward_custom = False

        self.compare_custom_and_cuda(batch_size, length, input_size, hidden_size, num_layers, bias,
                                     inter_layer_dropout, recurrent_dropout, batch_first, skip_connection,
                                     jit_forward_custom)

    def test_lstm_basic_multi_layer_length_first(self):
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
        batch_first = False
        jit_forward_custom = False

        self.compare_custom_and_cuda(batch_size, length, input_size, hidden_size, num_layers, bias,
                                     inter_layer_dropout, recurrent_dropout, batch_first, skip_connection,
                                     jit_forward_custom)

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
        batch_first = True
        skip_connection = False
        jit_forward_custom = False

        self.compare_custom_and_cuda(batch_size, length, input_size, hidden_size, num_layers, bias,
                                     inter_layer_dropout, recurrent_dropout, batch_first, skip_connection,
                                     jit_forward_custom)

    def test_lstm_basic_single_layer_jit(self):
        self.set_seed()
        # :param input: batch size x length x input_size
        # :param hx: num_layers x batch_size x hidden_size

        batch_size = 13
        length = 4
        input_size = 32
        hidden_size = 7
        num_layers = 1
        bias = True
        inter_layer_dropout = 0.
        recurrent_dropout = 0.
        batch_first = True
        skip_connection = False
        jit_forward_custom = True

        self.compare_custom_and_cuda(batch_size, length, input_size, hidden_size, num_layers, bias,
                                     inter_layer_dropout, recurrent_dropout, batch_first, skip_connection,
                                     jit_forward_custom)

    @staticmethod
    def compare_custom_and_cuda(batch_size, length, input_size, hidden_size, num_layers, bias,
                                inter_layer_dropout, recurrent_dropout, batch_first, skip_connection,
                                jit_forward_custom):
        if batch_first:
            input = torch.rand(batch_size, length, input_size)
        else:
            input = torch.rand(length, batch_size, input_size)

        hx = torch.rand(num_layers, batch_size, hidden_size), torch.rand(num_layers, batch_size, hidden_size)

        torch_lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                   num_layers=num_layers, bias=bias, dropout=inter_layer_dropout,
                                   batch_first=batch_first, bidirectional=False)

        custom_lstm = lstm.LSTM(input_size=input_size, hidden_size=hidden_size,
                                num_layers=num_layers, bias=bias,
                                inter_layer_dropout=inter_layer_dropout, recurrent_dropout=recurrent_dropout,
                                skip_connection=skip_connection, batch_first=batch_first)

        CustomLSTMTest.copy_weights_from_torch_to_custom(torch_lstm, custom_lstm, bias)

        if jit_forward_custom:
            custom_lstm_forward = torch.jit.script(custom_lstm)
        else:
            custom_lstm_forward = custom_lstm
        torch_output, torch_hidden = torch_lstm(input, hx)
        custom_output, custom_hidden = custom_lstm_forward(input, hx)

        torch_output.sum().backward()
        custom_output.sum().backward()

        torch_testing.assert_allclose(torch_hidden[0], custom_hidden[0])
        torch_testing.assert_allclose(torch_hidden[1], custom_hidden[1])

        torch_testing.assert_allclose(custom_output, torch_output)

        torch_grads = [p.grad for p in torch_lstm.parameters() if p.grad is not None]
        custom_grads = [p.grad for p in custom_lstm.parameters() if p.grad is not None]

        assert len(torch_grads) == len(custom_grads)

        for i in range(len(torch_grads)):
            torch_testing.assert_allclose(torch_grads[i], custom_grads[i])

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
