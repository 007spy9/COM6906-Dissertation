import tensorflow as tf
import numpy as np
import sklearn as sk
from sklearn.linear_model import Ridge
from tqdm import tqdm
import torch
import gc

from reservoir.DeepReadout import DeepReadout


class BasicESNCudaDeepReadout:
    def __init__(self, leakage_rate, spectral_radius, gamma, n_neurons, W_in, sparsity, class_weights=None, is_optimising=False, seed=None):
        print(tf.config.list_physical_devices('GPU'))
        print(f"Is CUDA available: {torch.cuda.is_available()}")

        self.is_cuda = torch.cuda.is_available()

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.leakage_rate = torch.tensor(leakage_rate, dtype=torch.float32, device='cuda')
        self.spectral_radius = torch.tensor(spectral_radius, dtype=torch.float32, device='cuda')
        self.gamma = torch.tensor(gamma, dtype=torch.float32, device='cuda')
        self.N = n_neurons
        self.W_in = torch.tensor(W_in, dtype=torch.float32, device='cuda')
        self.sparsity = sparsity

        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32, device='cuda')
        else:
            self.class_weights = None

        self.is_optimising = is_optimising

        # Generate the reservoir weights and apply a sparsity mask
        mask = np.random.choice([0, 1], size=(self.N, self.N), p=[1 - self.sparsity, self.sparsity])
        temp_W_res = np.random.uniform(-1, 1, (self.N, self.N))
        temp_W_res *= mask

        self.W_res = torch.tensor(temp_W_res, dtype=torch.float32, device='cuda')

        res_eigenvalues = torch.linalg.eigvals(self.W_res)

        # Scale the reservoir weights
        self.W_res /= torch.max(torch.abs(res_eigenvalues))

        # Our readout will be a dense layer, which then feeds into a softmax layer
        self.readout = None

        self.previous_states = []

        self.bar_update_step = 10000

        print(f"BasicESN initialised with leakage_rate: {leakage_rate}, spectral_radius: {spectral_radius}, gamma: {gamma}, n_neurons: {n_neurons}, sparsity: {sparsity}")

    def recurrent_unit(self, x, pbar=None):
        #state = np.zeros((self.N, 1))

        # for i in range(x.shape[0]):
        #     # Compute the reservoir state using the equation
        #     # h(t+1) = h(t) * leakage_rate  + (1 - leakage_rate) * tanh((gamma * W_in * x(t)) + (spectral_radius * W_res * h(t)) + bias)
        #     # non_linear = np.tanh(
        #     #     (self.gamma * self.W_in @ x[i].reshape(-1, 1)) + (self.spectral_radius * self.W_res @ state))
        #
        #     non_linear = torch.tanh(
        #         (self.gamma * self.W_in @ x[i].reshape(-1, 1)) + (self.spectral_radius * self.W_res @ state))
        #
        #     #state = (state * self.leakage_rate) + ((1 - self.leakage_rate) * non_linear)
        #     state = tf.math.add(tf.math.multiply(state, self.leakage_rate), tf.math.multiply((1 - self.leakage_rate), non_linear))
        #
        #     # Append the state after ravel
        #     # Tensorflow does not support the ravel method, so we will use numpy
        #     self.previous_states.append(tf.experimental.numpy.ravel(state))
        #
        #     if pbar:
        #         pbar.update(1)

        # x is in the shape (n_chunks, chunk_size, n_features)

        # Let's rewrite the above code using a cuda implementation
        state = torch.zeros((self.N, 1), device='cuda')

        previous_states_cuda = torch.zeros((x.shape[0], self.N), device='cuda')
        #previous_states = np.zeros((x.shape[0], self.N))

        x_cuda = torch.tensor(x, dtype=torch.float32, device='cuda')

        with torch.cuda.device(0):
            with torch.no_grad():
                for i in range(x.shape[0]):
                    non_linear = torch.tanh(
                        (self.gamma * torch.matmul(self.W_in, x_cuda[i].reshape(-1, 1))) + (self.spectral_radius * torch.matmul(self.W_res, state)))

                    state = torch.add(torch.mul(state, self.leakage_rate), torch.mul((1 - self.leakage_rate), non_linear))

                    previous_states_cuda[i] = state.ravel()

                    if pbar:
                        # Update every 10000 steps

                        if i % self.bar_update_step == 0:
                            #  If i is within the last bar_update_step steps, update the progress bar by the remaining steps
                            if i >= x.shape[0] - self.bar_update_step:
                                pbar.update(x.shape[0] - i)
                            else:
                                pbar.update(self.bar_update_step)

        previous_states = previous_states_cuda#.cpu()#.numpy()

        del previous_states_cuda
        del state
        del x_cuda

        gc.collect()
        torch.cuda.empty_cache()

        return previous_states

    def compute_reservoir_state(self, x):
        # Initialize the reservoir state
        # Define a previous_states list to store the state of the reservoir at each time step
        previous_states = []

        # If we are optimising, we will use the tqdm progress bar
        if not self.is_optimising:
            with tqdm(total=x.shape[0]) as pbar:
                previous_states = self.recurrent_unit(x, pbar)
        else:
            previous_states = self.recurrent_unit(x)

        # Flatten the previous_states list to a 2D array of shape (n_samples, n_neurons)
        #previous_states = np.array(previous_states)
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
        N_out_midpoint = (self.N + y.shape[1]) // 2
        readout = DeepReadout(self.N, N_out_midpoint, y.shape[1])

        # Create Tensors for the input and output data
        x_tensor = torch.tensor(x, dtype=torch.float32, device='cuda')
        y_tensor = torch.tensor(y, dtype=torch.float32, device='cuda')
        x_val_tensor = torch.tensor(x_val, dtype=torch.float32, device='cuda')
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32, device='cuda')

        # Compute the reservoir state
        state = self.compute_reservoir_state(x_tensor)

        # Put the state on the GPU
        state = torch.tensor(state, dtype=torch.float32, device='cuda')

        print("Reservoir state computed, fitting readout layer...")

        if x_val is not None and y_val is not None:
            print("Validation data provided, scoring readout layer based on validation data...")
            val_state = self.compute_reservoir_state(x_val_tensor)

            # Put the validation state on the GPU
            val_state = torch.tensor(val_state, dtype=torch.float32, device='cuda')

            # print(f"Shape of state before fitting: {state.shape}")
            # print(f"Shape of y before fitting: {y.shape}")

            # Fit the readout layer
            losses = readout.fit(state, y_tensor, x_val=val_state, y_val=y_val_tensor, class_weights=self.class_weights)
        else:
            losses = readout.fit(state, y_tensor, class_weights=self.class_weights)

        # Once the readout layer is fitted, or if no validation data is provided, fit the 'final' readout layer

        print("Readout layer fitted.")

        self.readout = readout

        return losses