import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.utils import compute_class_weight

from collect.DataframeCollector import DataframeCollector
from collect.TestSetSplitter import TestSetSplitter
from download.DataDownloader import DataDownloader

class DatasetPreparation:
    def __init__(self):
        self.is_kaggle = (os.environ.get("PWD", "") == "/kaggle/working")

        if self.is_kaggle:
            self.data_path = '/kaggle/input/har70'
        else:
            self.data_path = os.path.abspath(os.path.join('..', 'data'))

        self.test_set = None
        self.train_set = None
        self.validation_set = None

        self.train_df = None
        self.val_df = None
        self.test_df = None

        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None

        self.y_train_encoded = None
        self.y_val_encoded = None
        self.y_test_encoded = None

        self.y_train_decoded = None
        self.y_val_decoded = None
        self.y_test_decoded = None

        self.x_train_scaled = None
        self.x_val_scaled = None
        self.x_test_scaled = None

        self.encoder = None
        self.scaler = None

        self.class_weights = None

    def prepare_dataset(self, dataset_name, input_columns=None, output_columns=None):
        if input_columns is None:
            input_columns = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']
        if output_columns is None:
            output_columns = ['label']
        # --------------------------
        # Download Data
        # --------------------------
        if not self.is_kaggle:
            # Download data
            downloader = DataDownloader()
            downloader.download_data(dataset_name, self.data_path)


        # --------------------------
        # Load Data
        # --------------------------
        collector = DataframeCollector()
        collector.load_full_dataset(os.path.join(self.data_path, dataset_name))

        # The collector provides one large dataframe with all the data concatenated, and also an array of dataframes, one for each subject
        dataframes = collector.get_dataframes()
        full_data = collector.get_collected_data()

        # --------------------------
        # Split Data
        # --------------------------
        # Add record_num and timestamp columns + input_columns
        feature_cols = ['record_num', 'timestamp'] + input_columns
        features = full_data[feature_cols]
        labels = full_data[output_columns]

        test_split = 0.2
        train_split = 1 - test_split
        validation_split = 0.2

        validation_from_train = True
        use_slices = False

        splitter = TestSetSplitter(test_split=test_split, val_split=validation_split,
                                   val_from_train=validation_from_train, seed=1234)
        train, val, test = splitter.split(dataframes, use_slices=use_slices)

        slice_size = 128
        slices_in_df_1 = len(collector.get_dataframes()[0]) // slice_size

        print(f"Number of frames in training set: {len(train)}")
        print(f"Number of frames in validation set: {len(val)}")
        print(f"Number of frames in testing set: {len(test)}")

        self.train_set = train
        self.validation_set = val
        self.test_set = test

        # --------------------------
        # Join Data
        # --------------------------
        train_df = pd.concat(train)
        val_df = pd.concat(val)
        test_df = pd.concat(test)

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # --------------------------
        # X/Y Split
        # --------------------------
        x_train = train_df[input_columns].values
        y_train = train_df[output_columns].values.ravel()

        x_val = val_df[input_columns].values
        y_val = val_df[output_columns].values.ravel()

        x_test = test_df[input_columns].values
        y_test = test_df[output_columns].values.ravel()

        print(f"X_train shape: {x_train.shape}, Y_train shape: {y_train.shape}")
        print(f"X_val shape: {x_val.shape}, Y_val shape: {y_val.shape}")
        print(f"X_test shape: {x_test.shape}, Y_test shape: {y_test.shape}")

        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test

        # --------------------------
        # Encoding
        # --------------------------
        # Encode the labels
        encoder = OneHotEncoder(categories='auto')

        encoder.fit(y_train.reshape(-1, 1))

        y_train_encoded = encoder.transform(y_train.reshape(-1, 1))
        y_val_encoded = encoder.transform(y_val.reshape(-1, 1))
        y_test_encoded = encoder.transform(y_test.reshape(-1, 1))

        self.y_train_encoded = y_train_encoded.toarray()
        self.y_val_encoded = y_val_encoded.toarray()
        self.y_test_encoded = y_test_encoded.toarray()

        print(f"Y_train encoded shape: {y_train_encoded.shape}")
        print(f"Y_val encoded shape: {y_val_encoded.shape}")
        print(f"Y_test encoded shape: {y_test_encoded.shape}")

        y_train_decoded = encoder.inverse_transform(y_train_encoded)
        y_val_decoded = encoder.inverse_transform(y_val_encoded)
        y_test_decoded = encoder.inverse_transform(y_test_encoded)

        self.y_train_decoded = y_train_decoded
        self.y_val_decoded = y_val_decoded
        self.y_test_decoded = y_test_decoded

        print(f"Y_train decoded shape: {y_train_decoded.shape}")
        print(f"Y_val decoded shape: {y_val_decoded.shape}")
        print(f"Y_test decoded shape: {y_test_decoded.shape}")

        self.encoder = encoder

        # --------------------------
        # Normalisation
        # --------------------------
        scaler = MinMaxScaler(feature_range=(-1, 1))

        scaler.fit(x_train)

        x_train_scaled = scaler.transform(x_train)
        x_val_scaled = scaler.transform(x_val)
        x_test_scaled = scaler.transform(x_test)

        self.x_train_scaled = x_train_scaled
        self.x_val_scaled = x_val_scaled
        self.x_test_scaled = x_test_scaled

        print(f"X_train scaled shape: {x_train_scaled.shape}")
        print(f"X_val scaled shape: {x_val_scaled.shape}")
        print(f"X_test scaled shape: {x_test_scaled.shape}")

        self.scaler = scaler

        # --------------------------
        # Class Weights
        # --------------------------
        dataset = collector.get_collected_data()

        labels = dataset['label'].values

        # Calculate the class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)

        # Print the class weight for each class and the labels

        weight_map = {label: weight for label, weight in zip(np.unique(labels), class_weights)}

        for label, weight in weight_map.items():
            print(f'Class: {label}, Weight: {weight}')

        self.class_weights = class_weights

        print("Dataset preparation complete.")

    def get_sets(self):
        return self.train_set, self.validation_set, self.test_set

    def get_set_dataframes(self):
        return self.train_df, self.val_df, self.test_df

    def get_raw_data(self, set='train'):
        if set == 'train':
            return self.x_train, self.y_train
        elif set == 'val':
            return self.x_val, self.y_val
        elif set == 'test':
            return self.x_test, self.y_test

    def get_preprocessed_data(self, set='train'):
        if set == 'train':
            return self.x_train_scaled, self.y_train_encoded
        elif set == 'val':
            return self.x_val_scaled, self.y_val_encoded
        elif set == 'test':
            return self.x_test_scaled, self.y_test_encoded

    def get_decoded_labels(self, set='train'):
        if set == 'train':
            return self.y_train_decoded
        elif set == 'val':
            return self.y_val_decoded
        elif set == 'test':
            return self.y_test_decoded

    def get_encoder(self):
        return self.encoder

    def get_scaler(self):
        return self.scaler

    def get_class_weights(self):
        return self.class_weights