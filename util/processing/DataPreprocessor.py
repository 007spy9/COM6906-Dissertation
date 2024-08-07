import numpy as np


class DataPreprocessor:
    def __init__(self):
        pass

    def buffered_windows(self, window_size, x_data, y_data):
        '''
        Buffer the data into windows of a certain size.
        :param window_size: The size of the window. How many samples to include in each window
        :param x_data: The features that are to be buffered
        :param y_data: The labels for the data
        :return: A tuple of the buffered x data and the labels (x_data, y_data)
        '''
        temp_x_data = x_data.copy()
        temp_y_data = y_data.copy()

        # The data may not exactly fit into the window size, so we will repeat the last sample until it fits
        n_samples = x_data.shape[0]
        n_features = x_data.shape[1]

        remainder = n_samples % window_size
        if remainder != 0:
            temp_x_data = np.concatenate(
                [x_data, np.repeat(x_data[-1][np.newaxis, :], window_size - remainder, axis=0)], axis=0)

        X_data_buffered = np.array(
            [temp_x_data[i:i + window_size].flatten() for i in range(0, temp_x_data.shape[0], window_size)])

        # The labels need to be reshaped as well, each label is one-hot encoded, so we will need to make sure each sample can be associated with a one-hot encoded label
        # The shape of the labels is (n_samples, n_classes), we will reshape it to (n_samples//window_size, n_classes * window_size)

        # Similar to the features, we will repeat the last label until it fits
        n_samples = temp_y_data.shape[0]
        n_classes = temp_x_data.shape[1]

        remainder = n_samples % window_size
        if remainder != 0:
            temp_y_data = np.concatenate(
                [temp_y_data, np.repeat(temp_y_data[-1][np.newaxis, :], window_size - remainder, axis=0)], axis=0)

        y_data_buffered = np.array(
            [temp_y_data[i:i + window_size].flatten() for i in range(0, temp_y_data.shape[0], window_size)])

        print(f"Buffered data shape: {X_data_buffered.shape}, {y_data_buffered.shape}")

        return X_data_buffered, y_data_buffered

    def exponential_moving_average(self, alpha, x_data, y_data):
        '''
        Apply an exponential moving average to the data.
        :param alpha: The smoothing factor, where alpha is applied to the current value and (1 - alpha) is applied to the previous value
        :param x_data: The x features that are to be smoothed
        :param y_data: The labels for the data
        :return: A tuple of the smoothed x data and the labels (x_data, y_data)
        '''
        temp_x_data = x_data.copy()
        temp_y_data = y_data.copy()

        # For each feature, we will apply the exponential moving average
        # The formula for the exponential moving average is: S_t = alpha * X_t + (1 - alpha) * S_{t-1}
        # Where S_t is the smoothed value, X_t is the original value, and alpha is the smoothing factor
        # Note that the first value of the smoothed data is the same as the original data

        for i in range(temp_x_data.shape[1]):
            temp_x_data[:, i] = alpha * temp_x_data[:, i] + (1 - alpha) * np.roll(temp_x_data[:, i], 1)

        X_data_smoothed = temp_x_data

        print(f"Smoothed data shape: {X_data_smoothed.shape}, {y_data.shape}")

        return X_data_smoothed, y_data

    def fourier_smoothing(self, data, threshold=2e4):
        '''
        Apply a fourier transform to the data to remove high frequency noise past a certain threshold.
        By default, the threshold is set to 2e4, but this may need to be lowered
        :param data: The data to be smoothed
        :param threshold: The threshold value
        :return: The smoothed data
        '''
        temp_data = np.array(data)

        # The fourier transform can be used to de-noise a signal by effectively removing the high frequency noise
        # This is essentially a low-pass filter

        # We need to carry this out on each feature separately as the fourier transform is applied to each feature
        # For each feature, we will apply the fourier transform
        fourier_dataset = np.zeros_like(temp_data)
        for i in range(temp_data.shape[1]):
            transformed_data = np.fft.rfft(temp_data[:, i])
            frequencies = np.fft.rfftfreq(temp_data.shape[0])
            transformed_data[frequencies > threshold] = 0
            smoothed_data = np.fft.irfft(transformed_data)
            # Check that the shape is the same, and if not, pad with zeros
            if smoothed_data.shape[0] < temp_data.shape[0]:
                smoothed_data = np.pad(smoothed_data, (0, temp_data.shape[0] - smoothed_data.shape[0]))
            fourier_dataset[:, i] = smoothed_data

        print(f"Fourier data shape: {fourier_dataset.shape}")

        return fourier_dataset

    def pipeline(self, step_names, x_data, y_data):
        '''
        Run a pipeline of data preprocessing steps. As defined in the step_names list. Where each step is a string representing the name of the method to be called.
        :param step_names: A list of strings representing the names of the methods to be called
        :return: A tuple of the processed x data and the labels (x_data, y_data)
        '''
        temp_x_data = x_data.copy()
        temp_y_data = y_data.copy()

        for step in step_names:
            print(f"Running step: {step}")
            if step == 'buffered_windows':
                temp_x_data, temp_y_data = self.buffered_windows(10, temp_x_data, temp_y_data)
            elif step == 'exponential_moving_average':
                temp_x_data, temp_y_data = self.exponential_moving_average(0.9, temp_x_data, temp_y_data)
            elif step == 'fourier_smoothing':
                temp_x_data = self.fourier_smoothing(temp_x_data)

        return temp_x_data, temp_y_data

    def get_available_steps(self):
        '''
        Returns a list of names of the available methods
        :return: A list of strings
        '''
        return ['buffered_windows', 'exponential_moving_average', 'fourier_smoothing']