import tensorflow as tf
import numpy as np
import sklearn as sk
from sklearn.linear_model import Ridge
from tqdm import tqdm

class HierarchyESN:
    def __init__(self, leakage_rate_1, spectral_radius_1, gamma_1, n_neurons_1, W_in_1, leakage_rate_2, spectral_radius_2, gamma_2, n_neurons_2, W_in_2):
        self.leakage_rate_1 = leakage_rate_1
        self.spectral_radius_1 = spectral_radius_1
        self.gamma_1 = gamma_1
        self.N_1 = n_neurons_1
        self.W_in_1 = W_in_1

        self.W_res_1 = np.random.uniform(-1, 1, (self.N_1, self.N_1))

        res_eigenvalues_1 = np.linalg.eigvals(self.W_res_1)

        # Scale the reservoir weights
        self.W_res_1 /= np.max(np.abs(res_eigenvalues_1))

        self.leakage_rate_2 = leakage_rate_2
        self.spectral_radius_2 = spectral_radius_2
        self.gamma_2 = gamma_2
        self.N_2 = n_neurons_2
        self.W_in_2 = W_in_2

        self.W_res_2 = np.random.uniform(-1, 1, (self.N_2, self.N_2))

        res_eigenvalues_2 = np.linalg.eigvals(self.W_res_2)

        # Scale the reservoir weights
        self.W_res_2 /= np.max(np.abs(res_eigenvalues_2))

        # Now, we also need to prepare the connectivity matrix between the two reservoirs
        # This matrix will be of shape (n_neurons_1, n_neurons_2), and will be randomly initialized from a uniform distribution
        self.W_12 = np.random.uniform(-1, 1, (self.N_1, self.N_2))

        # Again, we need to scale the connectivity matrix
        self.W_12 /= np.max(np.abs(np.linalg.eigvals(self.W_12)))

        # We will use another ridge regression model for the readout layer
        self.readout = Ridge(alpha=1.0)

    def compute_reservoir_state(self, x):
        # Initialize the reservoir state
        # Define a previous_states list to store the state of the reservoir at each time step
        self.previous_states = []
        state1 = np.zeros((self.N_1, 1))
        state2 = np.zeros((self.N_2, 1))
        state = np.zeros((self.N_1 + self.N_2, 1))

        with tqdm(total=x.shape[0]) as pbar:
            for i in range(x.shape[0]):
                # Compute the reservoir state. As this is a hierarchical ESN, we will have two reservoirs
                # The hidden state of the first reservoir will be computed as follows:
                # h_1(t+1) = h_1(t) * leakage_rate_1  + (1 - leakage_rate_1) * tanh((gamma_1 * W_in_1 * x(t)) + (spectral_radius_1 * W_res_1 * h_1(t)) + bias_1)

                # The hidden state of the second reservoir will also include components of the first reservoir, defined through the connectivity matrix W_12
                # h_2(t+1) = h_2(t) * leakage_rate_2  + (1 - leakage_rate_2) * tanh((gamma_2 * W_in_2 * x(t)) + (spectral_radius_2 * W_res_2 * h_2(t)) + (W_12 * h_1(t)) + bias_2)

                non_linear_1 = np.tanh((self.gamma_1 * self.W_in_1 @ x[i].reshape(-1, 1)) + (self.spectral_radius_1 * self.W_res_1 @ state1))
                non_linear_2 = np.tanh((self.gamma_2 * self.W_in_2 @ x[i].reshape(-1, 1)) + (self.spectral_radius_2 * self.W_res_2 @ state2) + (self.W_12 @ state1))

                state_1 = (state1 * self.leakage_rate_1) + ((1 - self.leakage_rate_1) * non_linear_1)
                state_2 = (state2 * self.leakage_rate_2) + ((1 - self.leakage_rate_2) * non_linear_2)

                # The state of the second reservoir will be the output of the model concatenated with the state of the first reservoir
                state = np.concatenate((state_1, state_2), axis=0)

                # Append the state after ravel
                self.previous_states.append(state.ravel())

                pbar.update(1)

        # Flatten the previous_states list to a 2D array of shape (n_samples, n_neurons)
        self.previous_states = np.array(self.previous_states)
        print(f"Shape of previous_states: {self.previous_states.shape}")

        return self.previous_states

    def forward(self, x):
        # Initialize the output
        y = tf.zeros((1, 1), dtype=tf.float32)

        # Compute the reservoir state
        state = self.compute_reservoir_state(x)

        # Compute the output from the readout layer
        y = self.readout.predict(state)

        #y_pred = np.argmax(y, axis=1)

        return y

    def get_state_history(self):
        return self.previous_states

    def fit(self, x, y):
        # Compute the reservoir state
        state = self.compute_reservoir_state(x)

        print(f"Shape of state before fitting: {state.shape}")
        print(f"Shape of y before fitting: {y.shape}")

        # Fit the readout layer
        # TODO: Add class weights
        self.readout.fit(state, y)
