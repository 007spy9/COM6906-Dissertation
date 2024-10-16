import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import numpy as np
import sklearn as sk


#https://github.com/stefanonardo/pytorch-esn
# Accessed (2024-07-08)

class RecurrentUnit():
    '''
    Recurrent unit for the reservoir computing model.
    In this instance, the reservoir unit is a simple RNN cell.
    '''

    def __init__(self, leakage_rate, spectral_radius, gamma, n_neurons, W_in):
        self.leakage_rate = tf.Variable(leakage_rate, trainable=False, dtype=tf.float32)
        self.spectral_radius = spectral_radius
        self.gamma = gamma
        self.N = n_neurons
        self.W_in = W_in

        self.W_res = np.random.uniform(-1, 1, (self.N, self.N))

        res_eigenvalues = np.linalg.eigvals(self.W_res)

        # Scale the reservoir weights
        self.W_res /= np.max(np.abs(res_eigenvalues))

        self.previous_states = []

    def compute_reservoir_state(self, x):
        # Initialize the reservoir state
        # Define a previous_states list to store the state of the reservoir at each time step
        self.previous_states = []
        state = np.zeros((self.N, 1))

        for i in range(x.shape[0]):
            # Compute the reservoir state using the equation
            # h(t+1) = h(t) * leakage_rate  + (1 - leakage_rate) * tanh((gamma * W_in * x(t)) + (spectral_radius * W_res * h(t)) + bias)
            non_linear = np.tanh((self.gamma * self.W_in * x[i]) + (self.spectral_radius * self.W_res @ state))

            state = (state * self.leakage_rate) + ((1 - self.leakage_rate) * non_linear)

            self.previous_states.append(state)

        # Flatten the previous_states list to a 2D array of shape (n_samples, n_neurons)
        self.previous_states = np.array(self.previous_states).reshape(x.shape[0], self.N)

        return self.previous_states

    def get_state_history(self):
        return self.previous_states