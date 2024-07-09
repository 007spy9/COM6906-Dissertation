import torch
import torch.nn as nn
import torch.nn.functional as F


#https://github.com/stefanonardo/pytorch-esn
# Accessed (2024-07-08)

class RecurrentUnit(nn.Module):
    '''
    Recurrent unit for the reservoir computing model.
    In this instance, the reservoir unit is a simple RNN cell.
    '''

    def __init__(self, input_size, hidden_size, output_size, spectral_radius=0.9, leakage_rate=0.3):
        super(RecurrentUnit, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.spectral_radius = spectral_radius
        self.leakage_rate = leakage_rate

        self.activation = 'tanh'

        self.input_weights = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.reservoir_weights = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.output_weights = nn.Parameter(torch.Tensor(output_size, hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.input_weights, -1, 1)
        nn.init.uniform_(self.reservoir_weights, -1, 1)
        nn.init.uniform_(self.output_weights, -1, 1)

    def forward(self, input, hidden=None):
        # Compute the hidden state
        if hidden is None:
            hidden = torch.zeros(input.size(0), self.hidden_size)

        # Call a linear layer to map the input to the hidden state
        hidden = F.linear(input, self.input_weights, hidden)

        # Call the RNN cell to compute the hidden state
        hidden = nn.RNNCell(self.hidden_size, self.hidden_size)(hidden, self.reservoir_weights)

        # Call a linear layer to map the hidden state to the output
        hidden = F.linear(hidden, self.output_weights)

        return hidden

    def fit(self):
        # For now, there is no fitting to be done as the Basic ESN reservoir is not trainable
        pass

