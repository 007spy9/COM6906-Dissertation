import gc

import tensorflow as tf
import numpy as np
import sklearn as sk
from sklearn.linear_model import Ridge
import tqdm
import torch
from typing import Type

from reservoir.VariableReadout import VariableReadout


class ThreeHierarchyESNCudaVariableReadout:
    def __init__(self, leakage_rate_1, spectral_radius_1, gamma_1, n_neurons_1, W_in_1, sparsity_1, leakage_rate_2,
                 spectral_radius_2, gamma_2, n_neurons_2, W_in_2, sparsity_2,
                 leakage_rate_3, spectral_radius_3, gamma_3, n_neurons_3, W_in_3, sparsity_3,
                 class_weights=None, is_optimising=False, seed=None):
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
        self.sparsity_1 = sparsity_1

        mask_1 = np.random.choice([0, 1], size=(n_neurons_1, n_neurons_1), p=[1 - sparsity_1, sparsity_1])
        W_res_temp_1 = np.random.uniform(-1, 1, (n_neurons_1, n_neurons_1))
        W_res_temp_1 *= mask_1

        res_eigenvalues_1 = np.linalg.eigvals(W_res_temp_1)

        # Scale the reservoir weights
        self.W_res_1 = W_res_temp_1 / np.max(np.abs(res_eigenvalues_1))
        self.W_res_1 = torch.tensor(self.W_res_1, dtype=torch.float32, device='cuda')

        self.leakage_rate_2 = torch.tensor(leakage_rate_2, dtype=torch.float32, device='cuda')
        self.spectral_radius_2 = torch.tensor(spectral_radius_2, dtype=torch.float32, device='cuda')
        self.gamma_2 = torch.tensor(gamma_2, dtype=torch.float32, device='cuda')
        self.N_2 = n_neurons_2
        self.W_in_2 = torch.tensor(W_in_2, dtype=torch.float32, device='cuda')
        self.sparsity_2 = sparsity_2

        mask_2 = np.random.choice([0, 1], size=(n_neurons_2, n_neurons_2), p=[1 - sparsity_2, sparsity_2])
        W_res_temp_2 = np.random.uniform(-1, 1, (n_neurons_2, n_neurons_2))
        W_res_temp_2 *= mask_2

        res_eigenvalues_2 = np.linalg.eigvals(W_res_temp_2)

        # Scale the reservoir weights
        self.W_res_2 = W_res_temp_2 / np.max(np.abs(res_eigenvalues_2))
        self.W_res_2 = torch.tensor(self.W_res_2, dtype=torch.float32, device='cuda')

        self.leakage_rate_3 = torch.tensor(leakage_rate_3, dtype=torch.float32, device='cuda')
        self.spectral_radius_3 = torch.tensor(spectral_radius_3, dtype=torch.float32, device='cuda')
        self.gamma_3 = torch.tensor(gamma_3, dtype=torch.float32, device='cuda')
        self.N_3 = n_neurons_3
        self.W_in_3 = torch.tensor(W_in_3, dtype=torch.float32, device='cuda')
        self.sparsity_3 = sparsity_3

        mask_3 = np.random.choice([0, 1], size=(n_neurons_3, n_neurons_3), p=[1 - sparsity_3, sparsity_3])
        W_res_temp_3 = np.random.uniform(-1, 1, (n_neurons_3, n_neurons_3))
        W_res_temp_3 *= mask_3

        res_eigenvalues_3 = np.linalg.eigvals(W_res_temp_3)

        # Scale the reservoir weights
        self.W_res_3 = W_res_temp_3 / np.max(np.abs(res_eigenvalues_3))
        self.W_res_3 = torch.tensor(self.W_res_3, dtype=torch.float32, device='cuda')

        # Now, we also need to prepare the connectivity matrix between the two reservoirs
        # This matrix will be of shape (n_neurons_1, n_neurons_2), and will be randomly initialized from a uniform distribution
        W_12_temp = np.random.uniform(-1, 1, (n_neurons_2, n_neurons_1))

        # To ensure we have a square matrix, let's zero-fill the remaining elements
        zeros = np.zeros((max(n_neurons_1, n_neurons_2), max(n_neurons_1, n_neurons_2)))
        zeros[:n_neurons_2, :n_neurons_1] = W_12_temp

        # Again, we need to scale the connectivity matrix
        self.W_12 = W_12_temp / np.max(np.abs(np.linalg.eigvals(zeros)))
        self.W_12 = torch.tensor(self.W_12, dtype=torch.float32, device='cuda')

        W_23_temp = np.random.uniform(-1, 1, (n_neurons_3, n_neurons_2))

        # To ensure we have a square matrix, let's zero-fill the remaining elements
        zeros = np.zeros((max(n_neurons_2, n_neurons_3), max(n_neurons_2, n_neurons_3)))
        zeros[:n_neurons_3, :n_neurons_2] = W_23_temp

        # Again, we need to scale the connectivity matrix
        self.W_23 = W_23_temp / np.max(np.abs(np.linalg.eigvals(zeros)))
        self.W_23 = torch.tensor(self.W_23, dtype=torch.float32, device='cuda')

        # We will use another ridge regression model for the readout layer
        self.readout = None
        self.readout_layer = None

        self.previous_states = []

        # Let's add all the parameters to a props dictionary so that they can be passed to JIT scripts
        self.props = {
            'leakage_rate_1': self.leakage_rate_1,
            'spectral_radius_1': self.spectral_radius_1,
            'gamma_1': self.gamma_1,
            'N_1': self.N_1,
            'sparsity_1': self.sparsity_1,
            'leakage_rate_2': self.leakage_rate_2,
            'spectral_radius_2': self.spectral_radius_2,
            'gamma_2': self.gamma_2,
            'N_2': self.N_2,
            'sparsity_2': self.sparsity_2,
            'leakage_rate_3': self.leakage_rate_3,
            'spectral_radius_3': self.spectral_radius_3,
            'gamma_3': self.gamma_3,
            'N_3': self.N_3,
            'sparsity_3': self.sparsity_3,
            'W_in_1': self.W_in_1,
            'W_in_2': self.W_in_2,
            'W_in_3': self.W_in_3,
            'W_res_1': self.W_res_1,
            'W_res_2': self.W_res_2,
            'W_res_3': self.W_res_3,
            'W_12': self.W_12,
            'W_23': self.W_23,
            'class_weights': self.class_weights,
            'is_optimising': self.is_optimising,
            'bar_update_step': self.bar_update_step
        }

        self.losses = []
        self.val_losses = []

        print(
            f"HierarchyESN initialised with leakage_rate_1: {leakage_rate_1}, spectral_radius_1: {spectral_radius_1}, gamma_1: {gamma_1}, n_neurons_1: {n_neurons_1}, leakage_rate_2: {leakage_rate_2}, spectral_radius_2: {spectral_radius_2}, gamma_2: {gamma_2}, n_neurons_2: {n_neurons_2} and leakage_rate_3: {leakage_rate_3}, spectral_radius_3: {spectral_radius_3}, gamma_3: {gamma_3}, n_neurons_3: {n_neurons_3}")

    def set_readout_model(self, sequential_model):
        self.readout_layer = sequential_model

        # print(f"Shape of W_in_1: {W_in_1.shape}")
        # print(f"Shape of W_in_2: {W_in_2.shape}")
        # print(f"Shape of W_res_1: {self.W_res_1.shape}")
        # print(f"Shape of W_res_2: {self.W_res_2.shape}")
        # print(f"Shape of W_12: {self.W_12.shape}")

    def compute_reservoir_state(self, x):
        # Initialize the reservoir state
        # Define a previous_states list to store the state of the reservoir at each time step
        self.previous_states = []

        x = torch.tensor(x, dtype=torch.float32, device='cuda')

        if not self.is_optimising and False:
            with tqdm.tqdm(total=x.shape[0]) as pbar:
                previous_states = recurrent_unit(x=x, pbar=pbar, N_1=self.N_1, N_2=self.N_2, N_3=self.N_3,
                                                      spectral_radius_1=self.spectral_radius_1,
                                                      spectral_radius_2=self.spectral_radius_2,
                                                      spectral_radius_3=self.spectral_radius_3, gamma_1=self.gamma_1,
                                                      gamma_2=self.gamma_2, gamma_3=self.gamma_3,
                                                      leakage_rate_1=self.leakage_rate_1,
                                                      leakage_rate_2=self.leakage_rate_2,
                                                      leakage_rate_3=self.leakage_rate_3, W_in_1=self.W_in_1,
                                                      W_in_2=self.W_in_2, W_in_3=self.W_in_3, W_res_1=self.W_res_1,
                                                      W_res_2=self.W_res_2, W_res_3=self.W_res_3, W_12=self.W_12,
                                                      W_23=self.W_23, bar_update_step=self.bar_update_step)
        else:
            previous_states = recurrent_unit(x=x, N_1=self.N_1, N_2=self.N_2, N_3=self.N_3,
                                                  spectral_radius_1=self.spectral_radius_1,
                                                  spectral_radius_2=self.spectral_radius_2,
                                                  spectral_radius_3=self.spectral_radius_3, gamma_1=self.gamma_1,
                                                  gamma_2=self.gamma_2, gamma_3=self.gamma_3,
                                                  leakage_rate_1=self.leakage_rate_1,
                                                  leakage_rate_2=self.leakage_rate_2,
                                                  leakage_rate_3=self.leakage_rate_3, W_in_1=self.W_in_1,
                                                  W_in_2=self.W_in_2, W_in_3=self.W_in_3, W_res_1=self.W_res_1,
                                                  W_res_2=self.W_res_2, W_res_3=self.W_res_3, W_12=self.W_12,
                                                  W_23=self.W_23, bar_update_step=self.bar_update_step)

        # Flatten the previous_states list to a 2D array of shape (n_samples, n_neurons)
        #previous_states = np.array(previous_states)

        print(f"Shape of previous_states: {previous_states.shape}")

        return previous_states

    def forward(self, x):
        if self.readout_layer is None:
            raise ValueError("The readout layer has not been set. Please set the readout layer before calling the forward method.")

        # Initialize the output
        y = tf.zeros((1, 1), dtype=tf.float32)

        # Compute the reservoir state
        state = self.compute_reservoir_state(x)

        self.previous_states = state

        # Compute the output from the readout layer
        y = self.readout.predict(state)

        # y_pred = np.argmax(y, axis=1)

        return y

    def get_state_history(self):
        return self.previous_states

    def fit(self, x, y, x_val=None, y_val=None, epochs=100, batch_size=100):
        if self.readout_layer is None:
            raise ValueError("The readout layer has not been set. Please set the readout layer before calling the fit method.")

        readout = VariableReadout(self.readout_layer)

        # Create Tensors for the input and output data
        x_tensor = torch.tensor(x, dtype=torch.float32, device='cuda')
        y_tensor = torch.tensor(y, dtype=torch.float32, device='cuda')
        x_val_tensor = torch.tensor(x_val, dtype=torch.float32, device='cuda')
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32, device='cuda')

        state = self.compute_reservoir_state(x_tensor)

        # Put the state on the GPU
        state = torch.tensor(state, dtype=torch.float32, device='cuda')

        print("Reservoir state computed, fitting readout layer...")

        if x_val is not None and y_val is not None:
            print("Validation data provided, fitting readout layer with validation data...")
            val_state = self.compute_reservoir_state(x_val_tensor)

            # Put the validation state on the GPU
            val_state = torch.tensor(val_state, dtype=torch.float32, device='cuda')

            losses, val_losses = readout.fit(state, y_tensor, x_val=val_state, y_val=y_val_tensor, class_weights=self.class_weights, epochs=epochs, batch_size=batch_size)

        else:
            losses, val_losses = readout.fit(state, y_tensor, class_weights=self.class_weights, epochs=epochs, batch_size=batch_size)

        # Once the readout layer has been fitted, we can fit the readout layer with the training data

        print("Readout layer fitted.")

        self.readout = readout
        self.losses = losses
        self.val_losses = val_losses

    def get_losses(self):
        return self.losses, self.val_losses



@torch.jit.script
def recurrent_unit(x, N_1:int, N_2:int, N_3:int, spectral_radius_1, spectral_radius_2, spectral_radius_3, gamma_1,
                   gamma_2, gamma_3, leakage_rate_1, leakage_rate_2, leakage_rate_3, W_in_1, W_in_2, W_in_3,
                   W_res_1, W_res_2, W_res_3, W_12, W_23, bar_update_step:int):

    n_samples, n_features = x.shape

    state_1 = torch.zeros(n_samples, N_1, device='cuda')
    state_2 = torch.zeros(n_samples, N_2, device='cuda')
    state_3 = torch.zeros(n_samples, N_3, device='cuda')

    # Let's pre-allocate the previous_states list to store the state of the reservoir at each time step
    previous_states_cuda = []

    x_cuda = x



    with torch.cuda.device(0):
        with torch.no_grad():
            for i in range(n_samples):
                # Compute the reservoir state. As this is a hierarchical ESN, we will have two reservoirs
                # The hidden state of the first reservoir will be computed as follows:
                # h_1(t+1) = h_1(t) * leakage_rate_1  + (1 - leakage_rate_1) * tanh((gamma_1 * W_in_1 * x(t)) + (spectral_radius_1 * W_res_1 * h_1(t)) + bias_1)

                # The hidden state of the second reservoir will also include components of the first reservoir, defined through the connectivity matrix W_12
                # h_2(t+1) = h_2(t) * leakage_rate_2  + (1 - leakage_rate_2) * tanh((gamma_2 * W_in_2 * x(t)) + (spectral_radius_2 * W_res_2 * h_2(t)) + (W_12 * h_1(t)) + bias_2)

                # The hidden state of the third reservoir will also include components of the second reservoir, defined through the connectivity matrix W_23
                # h_3(t+1) = h_3(t) * leakage_rate_3  + (1 - leakage_rate_3) * tanh((gamma_3 * W_in_3 * x(t)) + (spectral_radius_3 * W_res_3 * h_3(t)) + (W_23 * h_2(t)) + bias_3)
                x_t = x_cuda[i].reshape(-1, 1)

                if i == 0:
                    # In this first step, the state_x[0] states need to be initialised without referring to a previous state
                    # This is because there is no previous state to refer to
                    # Therefore, only the input data needs to be applied to the non-linear function as the previous state is all zeros and so the state will be the same
                    non_linear_1 = torch.tanh(gamma_1 * torch.matmul(W_in_1, x_t))
                    non_linear_2 = torch.tanh(gamma_2 * torch.matmul(W_in_2, x_t))
                    non_linear_3 = torch.tanh(gamma_3 * torch.matmul(W_in_3, x_t))

                    state_1[i] = (1 - leakage_rate_1) * non_linear_1.squeeze(1)
                    state_2[i] = (1 - leakage_rate_2) * non_linear_2.squeeze(1)
                    state_3[i] = (1 - leakage_rate_3) * non_linear_3.squeeze(1)

                else:
                    # non_linear_1 = np.tanh((gamma_1 * W_in_1 @ x[i].reshape(-1, 1)) + (spectral_radius_1 * W_res_1 @ state1))
                    w_in_1_by_x = torch.matmul(W_in_1, x_t)
                    w_res_1_by_state_1 = torch.matmul(W_res_1, state_1[i - 1].reshape(-1, 1))

                    non_linear_1 = torch.tanh((gamma_1 * w_in_1_by_x) + (spectral_radius_1 * w_res_1_by_state_1))

                    # non_linear_2 = np.tanh((gamma_2 * W_in_2 @ x[i].reshape(-1, 1)) + (spectral_radius_2 * W_res_2 @ state2) + (W_12 @ state1))
                    w_in_2_by_x = torch.matmul(W_in_2, x_t)
                    w_res_2_by_state_2 = torch.matmul(W_res_2, state_2[i - 1].reshape(-1, 1))
                    # The shapes are (n_neurons_1, n_neurons_2) and (n_neurons_1, 1), so we need to transpose the second matrix
                    # An example is (400x100) and (1x400)
                    w_12_by_state_1 = torch.matmul(W_12, state_1[i - 1].reshape(-1, 1))
                    non_linear_2 = torch.tanh(
                        (gamma_2 * w_in_2_by_x) + (spectral_radius_2 * w_res_2_by_state_2) + w_12_by_state_1)

                    # non_linear_3 = np.tanh((gamma_3 * W_in_3 @ x[i].reshape(-1, 1)) + (spectral_radius_3 * W_res_3 @ state3) + (W_23 @ state2))
                    w_in_3_by_x = torch.matmul(W_in_3, x_t)
                    w_res_3_by_state_3 = torch.matmul(W_res_3, state_3[i - 1].reshape(-1, 1))

                    w_23_by_state_2 = torch.matmul(W_23, state_2[i - 1].reshape(-1, 1))
                    non_linear_3 = torch.tanh(
                        (gamma_3 * w_in_3_by_x) + (spectral_radius_3 * w_res_3_by_state_3) + w_23_by_state_2)

                    # state_1 = (state1 * leakage_rate_1) + ((1 - leakage_rate_1) * non_linear_1)
                    state_1[i] = ((leakage_rate_1 * state_1[i - 1].reshape(-1, 1)) + ((1 - leakage_rate_1) * non_linear_1)).squeeze(1)
                    # state_2 = (state2 * leakage_rate_2) + ((1 - leakage_rate_2) * non_linear_2)
                    state_2[i] = ((leakage_rate_2 * state_2[i - 1].reshape(-1, 1)) + ((1 - leakage_rate_2) * non_linear_2)).squeeze(1)
                    # state_3 = (state3 * leakage_rate_3) + ((1 - leakage_rate_3) * non_linear_3)
                    state_3[i] = ((leakage_rate_3 * state_3[i - 1].reshape(-1, 1)) + ((1 - leakage_rate_3) * non_linear_3)).squeeze(1)

                # The state of the second reservoir will be the output of the model concatenated with the state of the first reservoir
                state = torch.cat((state_1[i], state_2[i], state_3[i]), dim=0)

                # Append the state after ravel
                previous_states_cuda.append(state)

                # if pbar:
                #     if i % bar_update_step == 0:
                #         # If i is within the last bar_update_step steps, update the progress bar by the remaining steps
                #         if i >= x.shape[0] - bar_update_step:
                #             pbar.update(x.shape[0] - i)
                #         else:
                #             pbar.update(bar_update_step)
                if i % bar_update_step == 0:
                    if i >= n_samples - bar_update_step:
                        print(f"Step: {n_samples - i}/{n_samples}")
                    else:
                        print(f"Step: {i}/{n_samples}")


    previous_states = torch.stack(previous_states_cuda, dim=0)

    del previous_states_cuda
    del state_1
    del state_2
    del state_3
    del x_cuda

    #gc.collect()
    #torch.cuda.empty_cache()

    return previous_states
