import os
import pickle


class PickleCollector:
    def __init__(self):
        pass

    def variant_to_string(self, variant):
        # Convert the variant to a string as some variants are abbreviations

        # First, check if the variant contains underscores
        if '_' in variant:
            # If it does, we will split the variant into parts
            parts = variant.split('_')

            # Iterate through each part and call this function recursively
            for i, part in enumerate(parts):
                parts[i] = self.variant_to_string(part)

            # Join the parts back together
            return ' '.join(parts)

        else:
            # We will use a case statement to convert the variant to a string
            match variant:
                # A special case is if the variant is '', we will return 'Baseline'
                case '':
                    return 'Baseline'
                case 'baseline':
                    return 'Baseline'
                case 'augments':
                    return 'Best Augmentation'
                case 'fourier':
                    return 'Fourier Smoothing'
                case 'window':
                    return 'Buffered'
                case 'optimised':
                    return 'Optimised Hyperparameters'
                case 'deepReadout':
                    return 'Deep Readout'
                case 'spectrogram':
                    return 'Spectrogram'
                case 'ema':
                    return 'Exponential Moving Average'
                case 'variableReadout':
                    return 'Variable Readout'
                case 'ls':
                    return 'Linear-Softmax'
                case 'las':
                    return 'Linear-Activation-Softmax'
                case 'lals':
                    return 'Linear-Activation-Linear-Softmax'
                case 'ladls':
                    return 'Linear-Activation-Dropout-Linear-Softmax'
                case _:
                    return variant

    def dataset_to_id(self, dataset):
        match dataset:
            case 'har70':
                return 'har70plus'
            case _:
                return dataset

    def get_files(self, path):
        # In the provided file path, there will be a series of pickle files
        # Each pickle contains an ESN model that needs to be benchmarked
        # The name of the pickle is in the format '{modelType}_{dataset}_{variant}.pkl'
        # The modelType is either 'basicESN', hierarchyESN', or 'threeHierarchyESN'
        # There are also additional pickle files that contain other baselines, these can be ignored but logged in the console

        # The dataset is the name of the dataset that the model was trained on
        # The variant is a string that describes the model, for example 'augments', 'fourier', 'deepReadout', etc.

        # Verify the path exists
        if not os.path.exists(path):
            print(f"Path {path} does not exist.")
            return

        # Get all files in the path
        files = os.listdir(path)

        # Filter out the files that are not pickle files
        pickle_files = [f for f in files if f.endswith('.pkl')]

        # Iterate through each pickle file
        pickles = []
        for f in pickle_files:
            # Remove the .pkl extension
            file = f.replace('.pkl', '')

            # Split the file name

            # Note, sometimes a variant may not be present, this is a baseline attempt
            parts = file.split('_')

            # The model type is the first part of the file name
            modelType = parts[0]

            # The dataset is the second part of the file name
            dataset = parts[1]

            # The variant is the third part of the file name if it exists
            if len(parts) == 3:
                variant = parts[2]
            elif len(parts) == 2:
                variant = ''
            elif len(parts) > 3:
                variant = '_'.join(parts[2:])
            else:
                variant = ''

            properties = {
                'fileName': file,
                'filePath': os.path.join(path, f),
                'modelType': modelType,
                'dataset': self.dataset_to_id(dataset),
                'variant': self.variant_to_string(variant)
            }

            pickles.append(properties)

            # print(f"Model Type: {properties['modelType']}, Dataset: {properties['dataset']}, Variant: {properties['variant']}")

        return pickles

    def purge_unknown_pickles(self, pickles):
        known_model_types = ['basicESN', 'hierarchyESN', 'threeHierarchyESN']

        # Iterate through each pickle
        for p in pickles:
            # Check if the model type is known
            if p['modelType'] not in known_model_types:
                print(f"Unknown model type: {p['modelType']}")
                pickles.remove(p)
                continue

            # We have a test pickle on the mackeyglass dataset, we can ignore this too
            if p['dataset'] == 'mackeyglass':
                print(f"Ignoring mackeyglass dataset.")
                pickles.remove(p)
                continue

        return pickles

    def load_pickle(self, file_name):
        # Load the pickle file
        with open(file_name, 'rb') as f:
            model = pickle.load(f)

        return model