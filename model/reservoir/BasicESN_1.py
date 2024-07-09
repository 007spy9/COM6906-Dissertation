import tensorflow as tf
import numpy as np

class BasicESN_1:
    def __init__(self, leakage_rate, spectral_radius, N, W_in, T, T_concept):
        self.leakage_rate = tf.Variable(leakage_rate, trainable=False, dtype=tf.float32)
        self.spectral_radius = spectral_radius
        self.N = N
        self.W_in = W_in
        self.T = T
        self.T_concept = T_concept

        self.W_res = np.random.uniform(-1, 1, (N, N))

        # Scale the reservoir weights
        D = np.random.uniform(0,1,(N,N)) > np.ones((N,N)) * 1

        self.W_res = self.W_res * D.astype(int)

        res_eigenvalues = np.linalg.eigvals(self.W_res)

        # Scale the reservoir weights
        self.W_res /= np.max(np.abs(res_eigenvalues))
