import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn

from .RecurrentUnit import RecurrentUnit

#https://github.com/stefanonardo/pytorch-esn

class BasicESN(nn.Module):

    def __init__(self, input_size, reservoir_size, output_size, spectral_radius=0.9, leakage_rate=0.3):
        super(BasicESN, self).__init__()

        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size

        self.spectral_radius = spectral_radius
        self.leakage_rate = leakage_rate

        self.activation = 'tanh'

        self.input_weights = nn.Parameter(torch.randn(reservoir_size, input_size) * 0.1)
        self.reservoir_weights = nn.Parameter(torch.randn(reservoir_size, reservoir_size) * 0.1)
        self.output_weights = nn.Parameter(torch.randn(output_size, reservoir_size) * 0.1)

        #To start, the readout layer will be a linear layer
        self.readout = nn.Linear(reservoir_size, output_size)

        self.reservoir = RecurrentUnit(input_size, reservoir_size, output_size, spectral_radius, leakage_rate)

    def fit(self, input, target):
        # The only fitting to be done is for the linear readout layer
        # This requires a closed-form solution of linear regression
        # This can be solved by minimising the mean squared error between the target and the output
        # This can be done using the normal equation
        W = np.dot(np.linalg.pinv(input), target)
        self.readout.weight = nn.Parameter(torch.Tensor(W))

    def forward(self, input, hidden=None):
        with torch.no_grad():
            # Call the reservoir to compute the hidden state
            hidden = self.reservoir(input, hidden)

            with torch.enable_grad():
                # Call the readout layer to map the hidden state to the output
                hidden = self.readout(hidden)

        return hidden

