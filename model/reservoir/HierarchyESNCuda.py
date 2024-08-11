import gc

import tensorflow as tf
import numpy as np
import sklearn as sk
from sklearn.linear_model import Ridge
from tqdm import tqdm
import torch

class HierarchyESNCuda:
    def __init__(self, leakage_rate_1, spectral_radius_1, gamma_1, n_neurons_1, W_in_1, leakage_rate_2, spectral_radius_2, gamma_2, n_neurons_2, W_in_2, class_weights=None, is_optimising=False, seed=None):
        print(tf.config.list_physical_devices('GPU'))
        print(f"Is CUDA available: {torch.cuda.is_available()}")

        self.is_cuda = torch.cuda.is_available()

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.is_optimising = is_optimising

        self.bar_update_step = 10000

        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32, device='cuda')
        else:
            self.class_weights = None

        self.leakage_rate_1 = torch.tensor(leakage_rate_1, dtype=torch.float32, device='cuda')
        self.spectral_radius_1 = torch.tensor(spectral_radius_1, dtype=torch.float32, device='cuda')
        self.gamma_1 = torch.tensor(gamma_1, dtype=torch.float32, device='cuda')
        self.N_1 = n_neurons_1
        self.W_in_1 = torch.tensor(W_in_1, dtype=torch.float32, device='cuda')

        W_res_temp_1 = np.random.uniform(-1, 1, (n_neurons_1, n_neurons_1))

        res_eigenvalues_1 = np.linalg.eigvals(W_res_temp_1)

        # Scale the reservoir weights
        self.W_res_1 = W_res_temp_1 / np.max(np.abs(res_eigenvalues_1))
        self.W_res_1 = torch.tensor(self.W_res_1, dtype=torch.float32, device='cuda')


        self.leakage_rate_2 = torch.tensor(leakage_rate_2, dtype=torch.float32, device='cuda')
        self.spectral_radius_2 = torch.tensor(spectral_radius_2, dtype=torch.float32, device='cuda')
        self.gamma_2 = torch.tensor(gamma_2, dtype=torch.float32, device='cuda')
        self.N_2 = n_neurons_2
        self.W_in_2 = torch.tensor(W_in_2, dtype=torch.float32, device='cuda')

        W_res_temp_2 = np.random.uniform(-1, 1, (n_neurons_2, n_neurons_2))

        res_eigenvalues_2 = np.linalg.eigvals(W_res_temp_2)

        # Scale the reservoir weights
        self.W_res_2 = W_res_temp_2 / np.max(np.abs(res_eigenvalues_2))
        self.W_res_2 = torch.tensor(self.W_res_2, dtype=torch.float32, device='cuda')

        # Now, we also need to prepare the connectivity matrix between the two reservoirs
        # This matrix will be of shape (n_neurons_1, n_neurons_2), and will be randomly initialized from a uniform distribution
        W_12_temp = np.random.uniform(-1, 1, (n_neurons_2, n_neurons_1))

        # To ensure we have a square matrix, let's zero-fill the remaining elements
        zeros = np.zeros((max(n_neurons_1, n_neurons_2), max(n_neurons_1, n_neurons_2)))
        zeros[:n_neurons_2, :n_neurons_1] = W_12_temp

        # Again, we need to scale the connectivity matrix
        self.W_12 = W_12_temp / np.max(np.abs(np.linalg.eigvals(zeros)))
        self.W_12 = torch.tensor(self.W_12, dtype=torch.float32, device='cuda')

        # We will use another ridge regression model for the readout layer
        self.readout = Ridge(alpha=1.0)

        self.previous_states = []

        print(f"HierarchyESN initialised with leakage_rate_1: {leakage_rate_1}, spectral_radius_1: {spectral_radius_1}, gamma_1: {gamma_1}, n_neurons_1: {n_neurons_1}, leakage_rate_2: {leakage_rate_2}, spectral_radius_2: {spectral_radius_2}, gamma_2: {gamma_2}, n_neurons_2: {n_neurons_2}")

        # print(f"Shape of W_in_1: {W_in_1.shape}")
        # print(f"Shape of W_in_2: {W_in_2.shape}")
        # print(f"Shape of W_res_1: {self.W_res_1.shape}")
        # print(f"Shape of W_res_2: {self.W_res_2.shape}")
        # print(f"Shape of W_12: {self.W_12.shape}")

    def recurrent_unit(self, x, pbar=None):
        state_1 = torch.zeros((self.N_1, 1), device='cuda')
        state_2 = torch.zeros((self.N_2, 1), device='cuda')
        state = torch.zeros((self.N_1 + self.N_2, 1), device='cuda')

        previous_states_cuda = torch.zeros((x.shape[0], self.N_1 + self.N_2), device='cuda')
        previous_states = np.zeros((x.shape[0], self.N_1 + self.N_2))

        x_cuda = torch.tensor(x, dtype=torch.float32, device='cuda')

        with torch.cuda.device(0):
            with torch.no_grad():
                for i in range(x.shape[0]):
                    # Compute the reservoir state. As this is a hierarchical ESN, we will have two reservoirs
                    # The hidden state of the first reservoir will be computed as follows:
                    # h_1(t+1) = h_1(t) * leakage_rate_1  + (1 - leakage_rate_1) * tanh((gamma_1 * W_in_1 * x(t)) + (spectral_radius_1 * W_res_1 * h_1(t)) + bias_1)

                    # The hidden state of the second reservoir will also include components of the first reservoir, defined through the connectivity matrix W_12
                    # h_2(t+1) = h_2(t) * leakage_rate_2  + (1 - leakage_rate_2) * tanh((gamma_2 * W_in_2 * x(t)) + (spectral_radius_2 * W_res_2 * h_2(t)) + (W_12 * h_1(t)) + bias_2)

                    #non_linear_1 = np.tanh((self.gamma_1 * self.W_in_1 @ x[i].reshape(-1, 1)) + (self.spectral_radius_1 * self.W_res_1 @ state1))
                    w_in_1_by_x = torch.matmul(self.W_in_1, x_cuda[i].reshape(-1, 1))
                    w_res_1_by_state_1 = torch.matmul(self.W_res_1, state_1)

                    non_linear_1 = torch.tanh((torch.mul(self.gamma_1, w_in_1_by_x)) + (torch.mul(self.spectral_radius_1, w_res_1_by_state_1)))

                    #non_linear_2 = np.tanh((self.gamma_2 * self.W_in_2 @ x[i].reshape(-1, 1)) + (self.spectral_radius_2 * self.W_res_2 @ state2) + (self.W_12 @ state1))
                    w_in_2_by_x = torch.matmul(self.W_in_2, x_cuda[i].reshape(-1, 1))
                    w_res_2_by_state_2 = torch.matmul(self.W_res_2, state_2)
                    # The shapes are (n_neurons_1, n_neurons_2) and (n_neurons_1, 1), so we need to transpose the second matrix
                    # An example is (400x100) and (1x400)
                    w_12_by_state_1 = torch.matmul(self.W_12, state_1)
                    non_linear_2 = torch.tanh((torch.mul(self.gamma_2, w_in_2_by_x)) + (torch.mul(self.spectral_radius_2, w_res_2_by_state_2) + w_12_by_state_1))

                    #state_1 = (state1 * self.leakage_rate_1) + ((1 - self.leakage_rate_1) * non_linear_1)
                    state_1 = torch.add(torch.mul(state_1, self.leakage_rate_1), torch.mul((1 - self.leakage_rate_1), non_linear_1))
                    #state_2 = (state2 * self.leakage_rate_2) + ((1 - self.leakage_rate_2) * non_linear_2)
                    state_2 = torch.add(torch.mul(state_2, self.leakage_rate_2), torch.mul((1 - self.leakage_rate_2), non_linear_2))

                    # The state of the second reservoir will be the output of the model concatenated with the state of the first reservoir
                    state = torch.cat((state_1, state_2), 0)

                    # Append the state after ravel
                    previous_states_cuda[i] = state.ravel()

                    if pbar:
                        if i % self.bar_update_step == 0:
                            # If i is within the last bar_update_step steps, update the progress bar by the remaining steps
                            if i >= x.shape[0] - self.bar_update_step:
                                pbar.update(x.shape[0] - i)
                            else:
                                pbar.update(self.bar_update_step)

        previous_states = previous_states_cuda.cpu().numpy()

        del previous_states_cuda
        del state_1
        del state_2
        del state
        del x_cuda

        gc.collect()
        torch.cuda.empty_cache()

        return previous_states

    def compute_reservoir_state(self, x):
        # Initialize the reservoir state
        # Define a previous_states list to store the state of the reservoir at each time step
        self.previous_states = []

        if not self.is_optimising:
            with tqdm(total=x.shape[0]) as pbar:
                previous_states = self.recurrent_unit(x, pbar)
        else:
            previous_states = self.recurrent_unit(x)

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

    def fit(self, x, y, x_val=None, y_val=None):
        if self.class_weights is not None:
            class_weights = self.class_weights.cpu().numpy()

            # We need a list the same length as the number of samples with the class weights
            # y and y_val are one-hot encoded
            y_class = np.argmax(y, axis=1)
            y_val_class = np.argmax(y_val, axis=1)

            train_weights = [class_weights[i] for i in y_class]
            val_weights = [class_weights[i] for i in y_val_class]

        state = self.compute_reservoir_state(x)

        print("Reservoir state computed, fitting readout layer...")

        if x_val is not None and y_val is not None:
            print("Validation data provided, fitting readout layer with validation data...")
            val_state = self.compute_reservoir_state(x_val)

            alpha_vals = np.logspace(-5, 3, 10)

            scores = []

            for alpha in alpha_vals:
                temp_readout = Ridge(alpha=alpha)

                if self.class_weights is not None:
                    temp_readout.fit(state, y, sample_weight=train_weights)

                    score = temp_readout.score(val_state, y_val, sample_weight=val_weights)

                else:
                    temp_readout.fit(state, y)

                    score = temp_readout.score(val_state, y_val)

                scores.append(score)
                print(f"Alpha: {alpha}, Score: {score}")

            best_alpha = alpha_vals[np.argmax(scores)]

            print(f"Best alpha: {best_alpha}")

            self.readout = Ridge(alpha=best_alpha)

        # Once the readout layer has been fitted, we can fit the readout layer with the training data

        if self.class_weights is not None:
            self.readout.fit(state, y, sample_weight=train_weights)
        else:
            self.readout.fit(state, y)

        print("Readout layer fitted.")