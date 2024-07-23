import numpy as np

from sklearn.model_selection import train_test_split


class BinnedTestSetSplitter:
    def __init__(self, test_size, val_size, val_from_train=True, seed=None):
        self.test_size = test_size
        self.val_size = val_size
        self.val_from_train = val_from_train
        self.seed = seed

        if self.seed is not None:
            np.random.seed(self.seed)

    def generate_splits(self, features, labels, chunk_size=128):

        # The incoming data consists of all the features and labels

        # As we are working with time series data, the data should be split in a way that respects the time series nature
        # One way of doing this is reshaping the data into chunks of a fixed size
        # such that the shape of the data ends up as a 2D array of shape (n_chunks, (chunk_size x n_features))
        # This way, the time series nature of the data is preserved

        # The data begins as a dataframe, so it should be converted to a numpy array
        n_features = features.shape[1]
        features = features.to_numpy()
        labels = labels.to_numpy()

        # First, chunk the data into chunks of size chunk_size

        chunks = []
        label_chunks = []

        for i in range(0, len(features), chunk_size):
            # Check if the chunk is the last chunk
            if i + chunk_size > len(features):
                # If the chunk is the last chunk, then the chunk should be padded with the last value before the padding, not zeros
                padding = chunk_size - (len(features) % chunk_size)
                chunk = np.concatenate([features[i:], np.tile(features[-1], (padding, 1))])

                # The labels should also be padded with the last value before the padding
                label_chunk = np.concatenate([labels[i:], np.tile(labels[-1], padding)])
            else:
                chunk = features[i:i + chunk_size]
                label_chunk = labels[i:i + chunk_size]
            chunks.append(chunk)
            label_chunks.append(label_chunk)

        print("Chunks shape: ", np.array(chunks).shape)

        features = np.array(chunks)

        # Next, chunk the labels into chunks of size chunk_size

        # Reshape the features into shape (n_chunks, (chunk_size x n_features))
        features = features.reshape(-1, chunk_size * n_features)

        # Reshape the labels into chunks of size chunk_size
        labels = np.array(label_chunks).reshape(-1, chunk_size)

        print("Data shape: ", features.shape, labels.shape)

        # To stratify the data, the labels should be converted to a 1D array
        stratify_labels = labels[:, 0]

        # Split the dataset into training and testing sets
        train_features, test_features, train_labels, test_labels = train_test_split(features, labels,
                                                                                    test_size=self.test_size,
                                                                                    random_state=self.seed)
        # Split the training set into training and validation sets
        train_features, val_features, train_labels, val_labels = train_test_split(train_features, train_labels,
                                                                                  test_size=self.val_size,
                                                                                  random_state=self.seed)

        # Now that the data has been split into training, testing and validation sets, the data should be reshaped back into
        # the original shape

        # Reshape the features back into shape (n_chunks, chunk_size, n_features)
        train_features = train_features.reshape(-1, chunk_size, n_features)
        test_features = test_features.reshape(-1, chunk_size, n_features)
        val_features = val_features.reshape(-1, chunk_size, n_features)

        # Reshape the labels back into shape (n_chunks, chunk_size)
        train_labels = train_labels.reshape(-1, chunk_size)
        test_labels = test_labels.reshape(-1, chunk_size)
        val_labels = val_labels.reshape(-1, chunk_size)

        # Now we need to remove the chunking, so that the data is in the original shape of (n_samples, n_features)
        train_features = train_features.reshape(-1, n_features)
        test_features = test_features.reshape(-1, n_features)
        val_features = val_features.reshape(-1, n_features)

        # The labels should be reshaped into a 1D array of shape (n_samples,)
        train_labels = train_labels.flatten()
        test_labels = test_labels.flatten()
        val_labels = val_labels.flatten()

        self.test_x = test_features
        self.test_y = test_labels
        self.val_x = val_features
        self.val_y = val_labels
        self.train_x = train_features
        self.train_y = train_labels

    def get_splits(self) -> tuple:
        return self.train_x, self.train_y, self.val_x, self.val_y, self.test_x, self.test_y

    def get_test_set(self) -> tuple:
        return self.test_x, self.test_y

    def get_val_set(self) -> tuple:
        return self.val_x, self.val_y

    def get_train_set(self) -> tuple:
        return self.train_x, self.train_y

    def get_test_x(self):
        return self.test_x

    def get_test_y(self):
        return self.test_y

    def get_val_x(self):
        return self.val_x

    def get_val_y(self):
        return self.val_y

    def get_train_x(self):
        return self.train_x

    def get_train_y(self):
        return self.train_y