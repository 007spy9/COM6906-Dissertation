import tensorflow as tf
import numpy as np
import sklearn as sk

class BasicESN_1:
    def __init__(self, leakage_rate, spectral_radius, gamma, N, W_in):
        self.leakage_rate = tf.Variable(leakage_rate, trainable=False, dtype=tf.float32)
        self.spectral_radius = spectral_radius
        self.gamma = gamma
        self.N = N
        self.W_in = W_in

        self.W_res = np.random.uniform(-1, 1, (N, N))

        res_eigenvalues = np.linalg.eigvals(self.W_res)

        # Scale the reservoir weights
        self.W_res /= np.max(np.abs(res_eigenvalues))

        # To begin with, the readout layer is a Ridge from sklearn
        self.readout = sk.linear_model.Ridge(alpha=1.0)

    def compute_reservoir_state(self, x):
        # Initialize the reservoir state
        state = tf.zeros((self.N, 1), dtype=tf.float32)

        for i in range(x.shape[0]):
            # Compute the reservoir state using the equation
            # h(t+1) = h(t) * leakage_rate  + (1 - leakage_rate) * tanh((gamma * W_in * x(t)) + (spectral_radius * W_res * h(t)) + bias)
            non_linear = tf.tanh(
                (tf.matmul(self.W_in, x[i]) * self.gamma) + (tf.matmul(self.W_res, state) * self.spectral_radius))

            state = (state * self.leakage_rate) + ((1 - self.leakage_rate) * non_linear)

        return state

    def forward(self, x):
        # Initialize the output
        y = tf.zeros((1, 1), dtype=tf.float32)

        # Compute the reservoir state
        state = self.compute_reservoir_state(x)

        # Compute the output from the readout layer
        y = self.readout.predict(state)

        return y

    def fit(self, x, y):
        # Compute the reservoir state
        state = self.compute_reservoir_state(x)

        # Fit the readout layer
        # TODO: Add class weights
        self.readout.fit(state, y)
