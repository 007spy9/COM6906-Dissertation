import os

import pandas as pd


class DataframeCollector:
    def __init__(self):
        self.dataframes = []
        self.collected_data = None

    def load_full_dataset(self, dataset_folder):
        # All the datasets in use are in the form of a folder of csv files
        # Load all the csv files in the folder
        csv_files = [f for f in os.listdir(dataset_folder) if f.endswith('.csv')]

        print("Discovered ", len(csv_files), " csv files in ", dataset_folder)
        print("Loading the csv files into dataframes")

        for file in csv_files:
            df = pd.read_csv(os.path.join(dataset_folder, file))
            self.dataframes.append(df)

        print("Loaded ", len(self.dataframes), " dataframes")
        print("Concatenating the dataframes")

        # Once all the dataframes are loaded, concatenate them into a single dataframe
        self.collected_data = pd.concat(self.dataframes)
        return self.collected_data

    def get_collected_data(self):
        return self.collected_data

    def get_dataframes(self):
        return self.dataframes

    def get_dataframes_count(self):
        return len(self.dataframes)

    def get_collected_data_shape(self):
        if self.collected_data is None:
            return 0
        return self.collected_data.shape

    def get_dataframes_shape(self):
        if len(self.dataframes) == 0:
            return None
        return [df.shape for df in self.dataframes]

    def get_dataframes_columns(self):
        if len(self.dataframes) == 0:
            return None
        return [df.columns for df in self.dataframes]

    def clear_dataframes(self):
        self.dataframes = []
        self.collected_data = None

if __name__ == '__main__':
    collector = DataframeCollector()

    print("Testing with the HAR70+ dataset")
    collector.load_full_dataset('../../data/har70plus')
    print(collector.get_collected_data().head())
    print(collector.get_collected_data().shape)

    print(collector.get_dataframes_count())
    print(collector.get_dataframes_shape())
    print(collector.get_dataframes_columns())

    print("Clearing...")
    collector.clear_dataframes()

    print(collector.get_dataframes_count())
    print(collector.get_dataframes_shape())
    print(collector.get_dataframes_columns())
    print(collector.get_collected_data_shape())

    print("Testing with the HARTH dataset")
    collector.load_full_dataset('../../data/harth')
    print(collector.get_collected_data().head())
    print(collector.get_collected_data().shape)

    print(collector.get_dataframes_count())
    print(collector.get_dataframes_shape())
    print(collector.get_dataframes_columns())

    collector.clear_dataframes()