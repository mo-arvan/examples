from typing import Tuple, List, Optional

import torch
import torch.nn as nn

import sequence_dropout

"""
Implementation of LSTM model described in On the State of the Art of
Evaluation in Neural Language Models
https://arxiv.org/abs/1707.05589
"""


class LSTM(nn.Module):
    """
    The LSTM expects the input to be batch first: Batch size x length
    The LSTM does not support bidirectional as an argument
    """

    __constants__ = ["layers"]

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        inter_layer_dropout=0.1,
        recurrent_dropout=0.0,
        skip_connection=False,
        batch_first=False,
    ):
        super(LSTM, self).__init__()

        assert (
            num_layers >= 1 and inter_layer_dropout >= 0.0 and recurrent_dropout >= 0.0
        )

        self.layers = nn.ModuleList(
            [nn.LSTMCell(input_size=input_size, hidden_size=hidden_size, bias=bias)]
        )
        for layer in range(num_layers - 1):
            self.layers.append(
                nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size, bias=bias)
            )

        self.inter_layer_dropout = nn.Dropout(p=inter_layer_dropout)
        self.recurrent_dropout = sequence_dropout.SequenceDropout(p=recurrent_dropout)

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.skip_connection = skip_connection
        self.batch_first = batch_first

    def forward(
        self,
        input: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """

        :param input: batch size x length x input_size
        :param hx: num_layers x batch_size x hidden_size
        :return: batch size x length x input_size
        """
        if not self.batch_first:
            input = input.permute(1, 0, 2)
        input_length = input.shape[1]
        batch_size = input.shape[0]

        if hx is None:
            zeros = torch.zeros(
                self.num_layers,
                batch_size,
                self.hidden_size,
                dtype=input.dtype,
                device=input.device,
            )
            hx = (zeros, zeros)

        length_dim_tensor = hx[0][0]

        lstm_h_list: List[List[torch.Tensor]] = []
        lstm_c_list: List[List[torch.Tensor]] = []

        for i in range(input_length + 1):
            lstm_h_list.append([])
            lstm_c_list.append([])
            for __ in range(self.num_layers):
                lstm_h_list[i].append(torch.zeros_like(length_dim_tensor))
                lstm_c_list[i].append(torch.zeros_like(length_dim_tensor))

        lstm_h_list[0][0] = torch.ones_like(lstm_h_list[0][0])

        lstm_h_list[0] = hx[0].unbind(dim=0)
        lstm_c_list[0] = hx[1].unbind(dim=0)

        layer_index = -1
        for layer in self.layers:
            layer_index += 1
            self.recurrent_dropout.generate_new_mask()
            for index in range(input_length):
                if layer_index == 0:
                    lstm_input = input.select(dim=1, index=index)
                else:
                    lstm_input = lstm_h_list[index + 1][layer_index - 1]
                    lstm_input = self.inter_layer_dropout(lstm_input)
                h_0, c_0 = (
                    lstm_h_list[index][layer_index],
                    lstm_c_list[index][layer_index],
                )
                h_0, c_0 = self.recurrent_dropout(h_0), c_0  # self.recurrent_dropout(c_0)
                h_1, c_1 = layer(lstm_input, (h_0, c_0))
                lstm_h_list[index + 1][layer_index] = h_1.clone()
                lstm_c_list[index + 1][layer_index] = c_1.clone()

        lstm_h = torch.stack([torch.stack(l) for l in lstm_h_list])
        lstm_c = torch.stack([torch.stack(l) for l in lstm_c_list])

        output = lstm_h[1:]

        if self.skip_connection:
            output = output.sum(dim=1)
        else:
            output = output[:, -1, :, :]

        if self.batch_first:
            output = output.permute(1, 0, 2)
        return output, (lstm_h[-1], lstm_c[-1])
