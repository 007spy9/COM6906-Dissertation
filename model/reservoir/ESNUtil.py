import numpy as np


def generate_input_weights(n_neurons, n_features, density, method, seed=None):
    if seed is not None:
        np.random.seed(seed)

    if method == '0/1':
        input_weights = np.random.rand(n_neurons, n_features) < density
    elif method == 'normal':
        input_weights = np.random.normal(0, 1, (n_neurons, n_features))
    elif method == 'uniform':
        input_weights = np.random.uniform(-1, 1, (n_neurons, n_features))
    elif method == '-1/1':
        input_weights = np.random.uniform(-1, 1, (n_neurons, n_features)) < density

    return input_weights