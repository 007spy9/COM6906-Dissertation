import numpy as np
import pandas as pd


class TestSetSplitter:
    '''
    This class is responsible for splitting the incoming data into training and testing sets.
    This takes in an array of dataframes and splits the entire dataset into test, train and validation sets.
    '''

    def __init__(self, test_split=0.2, val_split=0.1, val_from_train=True, seed=None):
        '''
        Initialize the TestSetSplitter with the test and validation split ratios
        :param test_split: The ratio of data to be used for testing
        :param val_split: The ratio of data to be used for validation
        :param val_from_train: Boolean value indicating if the validation set should be taken from the training set or the entire dataset
        :param seed: The seed for the random number generator
        '''
        self.test_split = test_split
        self.val_split = val_split
        self.val_from_train = val_from_train
        self.seed = seed

        if self.seed is not None:
            np.random.seed(self.seed)

    def split(self, data, use_slices=True):
        '''
        Split the incoming data into training, testing and validation sets
        :param data: An array of dataframes to be split
        :param use_slices: Boolean value indicating if the data should be split into slices
        :return: A tuple of training, validation and testing sets, where each is an array of 128 length dataframes
        '''

        # Print the shape of the data
        print("Data shape: ", [df.shape for df in data])

        # Pick one of the dataframes in the data array to be the unseen subject
        unseen_subject = data[np.random.randint(0, len(data))]

        # Remove the unseen subject from the data
        data = [df for df in data if df is not unseen_subject]

        # As there are multiple dataframes, and the data is indexed by timestamp, it is possible that the dataframes
        # have overlapping timestamps. In order to account for this, a new index is created that will be unique for each
        # row in the concatenated dataframe.
        prev_max = 0
        total_data_points = 0
        for i, df in enumerate(data):
            number = np.arange(prev_max, prev_max + len(df))
            prev_max += len(df)
            df['record_num'] = number
            total_data_points += len(df)

        seen_data_points = total_data_points
        total_data_points += len(unseen_subject)
        # The structure of the data is such that each dataframe is a time series of data
        # The data should be split in a way that respects the time series nature
        # Therefore, the splitting strategy should be such that the time series nature is preserved
        # To do this, the data will be split on a per time series basis, with the test sets being taken from the end of each time series
        # The validation set will be taken from the end of the training set, if val_from_train is True, or in the same way as the test set if val_from_train is False
        # The training set will be the remaining data
        # The unseen subject will be added to the test set in order to include a full time series of unseen data

        # subject |                                  XX
        #         |                                  XX
        #         |                                  XX
        #         | xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxXX
        #         +------------------------------------+
        #                                         time
        # The approximate split is shown above, where the Xs represent the test set, and the rest is the training set
        # As can be seen, the test set is taken from the end of each time series, and the unseen subject is added to the test set

        train_data = []
        val_data = []
        test_data = []

        if use_slices:
            # The data is time series data, so the data should be split in a way that respects the time series nature
            # For our approach, the data will be split into slices of data, where each slice is a time series of a fixed length
            # The slices will be split into training, testing and validation sets

            # Our time slices will be of length 128
            slice_length = 128

            # Loop through the data and split it into slices
            for df in data:
                slices = []
                # Get the number of slices that can be taken from the dataframe
                n_slices = len(df) // slice_length

                # The dataframe might not be long enough to fill all the slices
                # If the dataframe is not long enough, then the data will be padded with zeros

                # Check if the dataframe is long enough to fill all the slices
                if len(df) % slice_length != 0:
                    # If the dataframe is not long enough, then the data will be padded with zeros
                    padding = slice_length - (len(df) % slice_length)
                    df = pd.concat([df, pd.DataFrame(np.zeros((padding, df.shape[1])), columns=df.columns)],
                                      ignore_index=True)
                    n_slices += 1

                # Split the dataframe into slices
                for i in range(n_slices):
                    slice = df.iloc[i * slice_length:(i + 1) * slice_length]
                    slices.append(slice)

                # Split the slices into training, testing and validation sets

                # The validation set will be taken from the training set if val_from_train is True, otherwise it will be taken from the entire dataset
                if self.val_from_train:
                    train_data.extend(slices[:-int(self.test_split * len(slices))])

                    # The validation set will be taken from the training set
                    train_len = len(train_data)

                    val_data.extend(train_data[-int(self.val_split * train_len):])

                    # The validation data should be removed from the training data
                    train_data = train_data[:-int(self.val_split * train_len)]
                else:
                    train_data.extend(slices[:-int(self.test_split * len(slices)) - int(self.val_split * len(slices))])

                    val_data.extend(slices[-int(self.test_split * len(slices)) - int(self.val_split * len(slices)):-int(
                        self.test_split * len(slices))])

                test_data.extend(slices[-int(self.test_split * len(slices)):])

            # The unseen subject will also be split into slices
            n_slices = len(unseen_subject) // slice_length

            # The unseen subject might not be long enough to fill all the slices
            # If the unseen subject is not long enough, then the data will be padded with zeros

            # Check if the unseen subject is long enough to fill all the slices
            if len(unseen_subject) % slice_length != 0:
                # If the unseen subject is not long enough, then the data will be padded with zeros
                padding = slice_length - (len(unseen_subject) % slice_length)
                unseen_subject = pd.concat(
                    [unseen_subject, pd.DataFrame(np.zeros((padding, unseen_subject.shape[1])),
                                                  columns=unseen_subject.columns)],
                    ignore_index=True)
                n_slices += 1

            # Split the unseen subject into slices
            for i in range(n_slices):
                slice = unseen_subject.iloc[i * slice_length:(i + 1) * slice_length]
                test_data.append(slice)


        else:
            # Iterate through the dataframes and split them into training, testing and validation sets

            for df in data:
                # Get the number of rows in the dataframe
                n_rows = len(df)

                # Split the dataframe into training, testing and validation sets


                if self.val_from_train:
                    train_data.append(df.iloc[:int((1 - self.test_split) * n_rows)])

                    # The validation set will be taken from the training set
                    train_len = len(train_data[-1])
                    val_data.append(train_data[-1].iloc[-int(self.val_split * train_len):])

                    train_data[-1] = train_data[-1].iloc[:-int(self.val_split * train_len)]
                else:
                    train_data.append(
                        df.iloc[:int((1 - self.test_split - self.val_split) * n_rows)])

                    val_data.append(
                        df.iloc[int((1 - self.test_split - self.val_split) * n_rows):int((1 - self.test_split) * n_rows)])

                test_data.append(df.iloc[int((1 - self.test_split) * n_rows):])

            # Append the unseen subject to the test set
            test_data.append(unseen_subject)

        return train_data, val_data, test_data
