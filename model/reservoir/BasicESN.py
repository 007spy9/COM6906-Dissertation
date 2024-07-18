import tensorflow as tf
import numpy as np
import sklearn as sk
from sklearn.linear_model import Ridge
from tqdm import tqdm

# This class is deprecated and is succeeded by the BasicESNCuda class
class BasicESN:
    def __init__(self, leakage_rate, spectral_radius, gamma, n_neurons, W_in, sparsity, is_optimising=False):
        '''
        Initialize the basic ESN model with the given parameters
        :param leakage_rate: Leakage rate of the reservoir (
        :param spectral_radius: Spectral radius of the reservoir weights
        :param gamma: Scaling factor for the input weights
        :param n_neurons: Number of neurons in the reservoir
        :param W_in: Input weights matrix (n_neurons, n_features)
        :param sparsity: Sparsity of the reservoir weights (where 1 is fully connected, and 0 is no connections)
        :param is_optimising: Whether the model is being optimised or not. Changes what logging is used
        '''
        self.leakage_rate = leakage_rate
        self.spectral_radius = spectral_radius
        self.gamma = gamma
        self.N = n_neurons
        self.W_in = W_in
        self.sparsity = sparsity

        self.is_optimising = is_optimising

        self.W_res = np.random.uniform(-1, 1, (self.N, self.N))

        # Sparsify the reservoir weights
        mask = np.random.choice([0, 1], size=(self.N, self.N), p=[1 - self.sparsity, self.sparsity])
        self.W_res *= mask

        res_eigenvalues = np.linalg.eigvals(self.W_res)

        # Scale the reservoir weights
        self.W_res /= np.max(np.abs(res_eigenvalues))

        # To begin with, the readout layer is a Ridge from sklearn
        self.readout = Ridge(alpha=1.0)

        self.previous_states = []

        print(f"BasicESN initialised with leakage_rate: {leakage_rate}, spectral_radius: {spectral_radius}, gamma: {gamma}, n_neurons: {n_neurons}, sparsity: {sparsity}")

    def recurrent_unit(self, x, previous_states, pbar=None):
        state = np.zeros((self.N, 1))

        for i in range(x.shape[0]):
            # Compute the reservoir state using the equation
            # h(t+1) = h(t) * leakage_rate  + (1 - leakage_rate) * tanh((gamma * W_in * x(t)) + (spectral_radius * W_res * h(t)) + bias)
            non_linear = np.tanh(
                (self.gamma * self.W_in @ x[i].reshape(-1, 1)) + (self.spectral_radius * self.W_res @ state))

            state = (state * self.leakage_rate) + ((1 - self.leakage_rate) * non_linear)

            # Append the state after ravel
            previous_states.append(state.ravel())

            if pbar:
                pbar.update(1)

        return state

    def compute_reservoir_state(self, x):
        # Initialize the reservoir state
        # Define a previous_states list to store the state of the reservoir at each time step
        previous_states = []

        # If we are optimising, we will use the tqdm progress bar
        if not self.is_optimising:
            with tqdm(total=x.shape[0]) as pbar:
                self.recurrent_unit(x, previous_states, pbar)
        else:
            self.recurrent_unit(x, previous_states)

        # Flatten the previous_states list to a 2D array of shape (n_samples, n_neurons)
        previous_states = np.array(previous_states)
        print(f"Shape of previous_states: {previous_states.shape}")

        return previous_states

    def forward(self, x):
        # Initialize the output
        y = tf.zeros((1, 1), dtype=tf.float32)

        # Compute the reservoir state
        state = self.compute_reservoir_state(x)

        self.previous_states = state

        # Compute the output from the readout layer
        y = self.readout.predict(state)

        #y_pred = np.argmax(y, axis=1)

        return y

    def get_state_history(self):
        return self.previous_states

    def fit(self, x, y, class_weights=None, x_val=None, y_val=None):
        # Compute the reservoir state
        state = self.compute_reservoir_state(x)

        if x_val is not None and y_val is not None:
            val_state = self.compute_reservoir_state(x_val)

            # print(f"Shape of state before fitting: {state.shape}")
            # print(f"Shape of y before fitting: {y.shape}")

            alpha_vals = np.logspace(0, 5, 10)

            scores = []

            for alpha in alpha_vals:
                temp_readout = Ridge(alpha=alpha)

                # Fit the readout layer
                temp_readout.fit(state, y)

                score = temp_readout.score(val_state, y_val)

                scores.append(score)

                print(f"Alpha: {alpha}, Score: {score}")

            best_alpha = alpha_vals[np.argmax(scores)]

            print(f"Best alpha: {best_alpha}")

            self.readout = Ridge(alpha=best_alpha)

            self.readout.fit(state, y)

        else:
            # Fit the readout layer
            self.readout.fit(state, y)