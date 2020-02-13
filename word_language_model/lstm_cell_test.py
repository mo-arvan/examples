import unittest

import torch.nn
import torch.testing as torch_testing

import lstm_cell


class LSTMCellTest(unittest.TestCase):
    def test_basic_lstm(self):
        input_size = 21
        hidden_size = 12
        bias = True
        batch_size = 4
        input = torch.rand((batch_size, input_size))
        hx = torch.rand(batch_size, hidden_size), torch.rand(batch_size, hidden_size)

        torch_lstm_cell = torch.nn.LSTMCell(input_size=input_size, hidden_size=hidden_size, bias=bias)
        custom_lstm_cell = lstm_cell.LSTMCell(input_size=input_size, hidden_size=hidden_size, cap_input_gate=False)

        self.copy_weights_from_torch_to_custom(torch_lstm_cell, custom_lstm_cell)

        torch_output = torch_lstm_cell(input, hx)
        custom_output = custom_lstm_cell(input, hx)

        torch_testing.assert_allclose(torch_output[0], custom_output[0])
        torch_testing.assert_allclose(torch_output[1], custom_output[1])

    @staticmethod
    def copy_weights_from_torch_to_custom(torch_model, custom_model, bias=True):
        parameter_list = list(torch_model.parameters())

        custom_model.weight_ih = parameter_list[0]
        custom_model.weight_hh = parameter_list[1]
        if bias:
            custom_model.bias = parameter_list[2]
            parameter_list[3].data.zero_()


if __name__ == '__main__':
    unittest.main()
