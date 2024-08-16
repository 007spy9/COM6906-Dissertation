import gc

import numpy as np
import librosa


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

        #print(f"X data shape: {x_data.shape}")
        #print(f"Y data shape: {y_data.shape}")

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
        #print(f"Y data shape: {temp_y_data.shape}")
        n_samples = temp_y_data.shape[0]
        n_classes = temp_y_data.shape[1]


        remainder = n_samples % window_size
        if remainder != 0:
            temp_y_data = np.concatenate([temp_y_data, np.repeat(temp_y_data[-1][np.newaxis, :], window_size - remainder, axis=0)], axis=0)


        y_data_buffered = np.array(
            [temp_y_data[i:i + window_size].flatten() for i in range(0, temp_y_data.shape[0], window_size)])

        print(f"Buffered data shape: {X_data_buffered.shape}, {y_data_buffered.shape}")

        gc.collect()

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

    def spectrogram_data(self, x_data, y_data, n_fft=2048, stride=512, reshape=True):
        '''
        Convert the data into a spectrogram representation.
        The n_fft and stride parameters should ideally be powers of 2 for efficiency.
        :param x_data: The features that are to be converted
        :param y_data: The labels for the data (one label per sample)
        :param n_fft: The number of samples to use for the fourier transform (window size)
        :param stride: The stride between each fourier transform in samples (hop length)
        :param reshape: Boolean value indicating if the data should be reshaped to a format that can be used for training or left as a 3D array
        :return: A tuple of the spectrogram data and the labels (x_data_spectrogram, y_data_spectrogram)
        '''

        temp_spectrograms = []

        max_time_frames = 0

        # Let's apply the fourier transform to each feature
        for i in range(x_data.shape[1]):
            spectrogram = np.abs(librosa.stft(x_data[:, i], n_fft=n_fft, hop_length=stride)) ** 2
            temp_spectrograms.append(spectrogram)
            max_time_frames = max(max_time_frames, spectrogram.shape[1])

        spectrogram_dataset = np.zeros((max_time_frames, n_fft // 2 + 1, x_data.shape[1]))

        for i, spectrogram in enumerate(temp_spectrograms):
            # Pad the spectrogram with zeros if the number of time frames is less than the maximum
            if spectrogram.shape[1] < max_time_frames:
                spectrogram = np.pad(spectrogram, ((0, 0), (0, max_time_frames - spectrogram.shape[1])))

            spectrogram_dataset[:spectrogram.shape[1], :, i] = spectrogram.T

        # We also need to adjust the labels to match the shape of the spectrogram data
        # As the labels are for each sample, we need to determine the label for each time frame in the spectrogram
        # We can do this by taking the most common label for each time frame
        # Each time frame is of length n_nfft // 2 + 1, and the labels for each frame will be at the index
        # x[i * stride : i * stride + n_fft // 2 + 1] where i is the time frame index

        # For the current x dataset, we will iterate through each time frame
        y_data_frames = np.zeros(spectrogram_dataset.shape[0])
        for t in range(len(spectrogram_dataset)):
            start = t * stride
            end = start + n_fft // 2 + 1
            labels = y_data[start:end]
            most_common_label = np.bincount(labels).argmax()
            y_data_frames[t] = most_common_label

        # As the spectrogram data is 3D, we need to reshape it to 2D
        # The shape of the spectrogram data is (n_frames, n_freq_bins, n_features)
        # We will reshape it to (n_frames, n_freq_bins * n_features)
        # This will allow us to train the model on the spectrogram data

        if reshape:
            n_frames = spectrogram_dataset.shape[0]
            n_freq_bins = spectrogram_dataset.shape[1]
            n_features = spectrogram_dataset.shape[2]
            spectrogram_dataset = spectrogram_dataset.reshape(n_frames, n_freq_bins * n_features)

        x_data_spectrogram = spectrogram_dataset

        y_data_spectrogram = y_data_frames

        print(f"Spectrogram data shape: {x_data_spectrogram.shape}, {y_data_spectrogram.shape}")

        return x_data_spectrogram, y_data_spectrogram

    def spectrogram_predictions_to_samples(self, y_pred, n_samples, n_fft, stride):
        '''
        Convert the one-hot encoded predictions from the spectrogram data back to the original samples using an accumulation method.
        Both input and output will be one-hot encoded.
        :param y_pred: The predictions from the model in the shape (n_frames, n_classes)
        :param n_samples: The number of samples in the original data (before the spectrogram transformation)
        :param n_fft: The window size from the spectrogram transformation
        :param stride: The stride from the spectrogram transformation
        :return: The one-hot encoded samples in the shape (n_samples, n_classes)
        '''
        n_frames, n_classes = y_pred.shape
        y_accumulations = np.zeros((n_samples, n_classes), dtype=float)
        y_counts = np.zeros(n_samples, dtype=int)

        for f in range(n_frames):
            start = f * stride
            end = start + n_fft

            # Bounds checking
            if end > n_samples:
                end = n_samples

            # Accumulate the predictions for the range[start:end]
            for i in range(start, end):
                y_accumulations[i] += y_pred[f]
                y_counts[i] += 1

        # Normalise the accumulator by contribution count
        # Avoid division by zero
        y_counts[y_counts == 0] = 1

        # Normalise the accumulator along the class axis
        y_accumulations /= y_counts[:, None]

        # Convert the accumulator to the most likely class for each sample
        y_samples = np.argmax(y_accumulations, axis=1)

        y_samples_onehot = np.zeros_like(y_accumulations)
        y_samples_onehot[np.arange(n_samples), y_samples] = 1

        print(f"Cumulative samples shape: {y_samples_onehot.shape}")

        return y_samples_onehot

    def pipeline(self, step_names, x_data, y_data, params):
        '''
        Run a pipeline of data preprocessing steps. As defined in the step_names list. Where each step is a string representing the name of the method to be called.
        :param step_names: A list of strings representing the names of the methods to be called
        :param x_data: The features that are to be processed
        :param y_data: The labels for the data
        :param params: A list of parameters to be passed to the methods (should be the same length as step_names)
        :return: A tuple of the processed x data and the labels (x_data, y_data)
        '''
        temp_x_data = x_data.copy()
        temp_y_data = y_data.copy()

        for i, step in enumerate(step_names):
            if step == 'buffered_windows':
                temp_x_data, temp_y_data = self.buffered_windows(params[i], temp_x_data, temp_y_data)
            elif step == 'exponential_moving_average':
                temp_x_data, temp_y_data = self.exponential_moving_average(params[i], temp_x_data, temp_y_data)
            elif step == 'fourier_smoothing':
                temp_x_data = self.fourier_smoothing(temp_x_data, params[i])

        return temp_x_data, temp_y_data

    def get_available_steps(self):
        '''
        Returns a list of names of the available methods
        :return: A list of strings
        '''
        return ['buffered_windows', 'exponential_moving_average', 'fourier_smoothing']