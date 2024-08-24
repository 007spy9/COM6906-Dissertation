import json
import os
import sys
import numpy as np
import pandas as pd
import sklearn
import tqdm
import torch
import tensorflow as tf
import librosa

from benchmarker import Benchmarker
from pickle_collector import PickleCollector

# We need to load the DatasetPreparation module from ../util/collect
# To do this, we need to add the path to the sys.path
util_path = os.path.abspath(os.path.join('..', 'util'))
models_path = os.path.abspath(os.path.join('..', 'model'))
sys.path.append(util_path)
sys.path.append(models_path)

from collect.DatasetPreparation import DatasetPreparation
from processing.DataPreprocessor import DataPreprocessor

loaded_datasets = {

}


def prepare_dataset(dataset_name, variant):
    dataset_prep = DatasetPreparation()
    data_preprocessor = DataPreprocessor()

    variant_lower = variant.lower()
    variant_slug = variant_lower.replace(' ', '_')

    requires_extra_processing = ['spectrogram', 'fourier', 'window', 'buffered', 'exponential moving average']

    # The dataset slug is a combination of the dataset name and the variant
    # If the variant requires extra processing, it should be included in the slug, otherwise it will just be the dataset name
    if any([req in variant_lower for req in requires_extra_processing]):
        dataset_slug = f"{dataset_name}_{variant_slug}"
    else:
        dataset_slug = dataset_name


    # Check if the dataset has already been loaded
    if dataset_slug in loaded_datasets:
        print(f"Dataset {dataset_slug} is cached. Returning cached dataset.")
        return loaded_datasets[dataset_slug]

    print(f"Preparing dataset {dataset_slug}...")
    # In order to prevent repeated data loading, we will load the dataset once and store it in memory

    input_features = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']
    output_features = ['label']

    dataset_prep.prepare_dataset(dataset_name, input_features, output_features)

    x_test, y_test = dataset_prep.get_preprocessed_data('test')

    if any([req in variant_lower for req in requires_extra_processing]):
        if 'spectrogram' in variant_lower:
            # Overwrite the test data with raw data as the spectrogram does not work with the preprocessed data
            x_test, y_test = dataset_prep.get_raw_data('test')
            x_data, y_data = data_preprocessor.spectrogram_data(x_data=x_test, y_data=y_test, n_fft=32, stride=32, encode_labels=True, label_encoder=dataset_prep.get_encoder())
        elif 'fourier' in variant_lower and ('window' in variant_lower or 'buffered' in variant_lower):
            x_data, y_data = data_preprocessor.pipeline(step_names=['fourier_smoothing', 'buffered_windows'], x_data=x_test, y_data=y_test, params=[100.0, 10])
        elif 'fourier' in variant_lower:
            x_data = data_preprocessor.fourier_smoothing(data=x_test, threshold=0.1)
            y_data = y_test
        elif 'window' in variant_lower or 'buffered' in variant_lower:
            x_data, y_data = data_preprocessor.buffered_windows(x_data=x_test, y_data=y_test, window_size=10)
        elif 'exponential moving average' in variant_lower:
            x_data, y_data = data_preprocessor.exponential_moving_average(x_data=x_test, y_data=y_test, alpha=0.999)

    else:
        x_data = x_test
        y_data = y_test

    # Store the dataset in the loaded_datasets dictionary
    loaded_datasets[dataset_slug] = (x_data, y_data)

    return x_data, y_data


def run_benchmarks(n_runs):
    print("Preparing for benchmarks...")

    results = []

    pickle_collector = PickleCollector()
    benchmarker = Benchmarker()

    # The files should be in the ../training directory
    path = '../training'

    # Convert the path to an absolute path
    path = os.path.abspath(path)

    print("Collecting models...")

    # Get the files in the path
    files = pickle_collector.get_files(path)

    # Purge the unknown pickles
    files = pickle_collector.purge_unknown_pickles(files)

    if files is None:
        print('There are no models to benchmark.')
        return

    # Iterate through each file
    for f in files:
        print(f"Model Type: {f['modelType']}, Dataset: {f['dataset']}, Variant: {f['variant']}")

    print("Running benchmarks...")

    # Iterate through each file
    for f in files:
        print(f"Running benchmark for {f['fileName']}...")

        # Load the pickle file
        print(f"Loading pickle file {f['fileName']}...")
        model = pickle_collector.load_pickle(f['filePath'])

        # Prepare the dataset
        print(f"Preparing dataset {f['dataset']}...")
        x_test, y_test = prepare_dataset(f['dataset'], f['variant'])

        # Run the benchmark
        results.append(benchmarker.run(f, model, x_test, y_test, n_runs))


    print('Benchmarks complete.')

    return results


if __name__ == '__main__':
    n_runs = 10
    results = run_benchmarks(n_runs)

    # Save the results to a JSON file
    with open('results.json', 'w') as f:
        json.dump(results, f)

