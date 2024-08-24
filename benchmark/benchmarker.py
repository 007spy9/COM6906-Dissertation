import os
import tracemalloc
import timeit

import json

import torch


class Benchmarker:
    def __init__(self):
        self.known_models = ['BasicESN', 'HierarchyESN', 'ThreeHierarchyESN']
        pass

    def run(self, model_properties, model, x_data, y_data, n_runs=10):
        # For each model, we will run a forward pass on the data and record the memory usage and time taken
        # To do this, we will use the tracemalloc module to record the memory usage
        # and the timeit module to record the time taken

        # The benchmark will be run multiple times to get an average and standard deviation

        # We will store the results in a dictionary with the following format:
        # results = {
        #     'n_runs': n_runs,
        #     'model_class': model,
        #     'modelType': modelType,
        #     'dataset': dataset,
        #     'variant': variant,
        #     'memory': {
        #         'average': average_memory,
        #         'std': std_memory,
        #         'results': memory_results
        #     },
        #     'time': {
        #         'average': average_time,
        #         'std': std_time,
        #         'results': time_results
        #     }
        # }

        # First, we will check if the model is a valid class, that is it is one of our expected models, and it has the forward method
        # Check the class of the model contains one of the known models. The model name may not be identical, but it should contain the known model name
        model_name = model.__class__.__name__
        if not any([m in model_name for m in self.known_models]):
            print(f"Model {model} is not a known model.")
            return

        if not hasattr(model, 'forward'):
            print(f"Model {model} does not have a forward method.")
            return

        # We will run the benchmark multiple times
        results = {
            'n_runs': n_runs,
            'model_class': model.__class__.__name__,
            'modelType': model_properties['modelType'],
            'dataset': model_properties['dataset'],
            'variant': model_properties['variant'],
            'memory_cpu': {
                'average': 0.0,
                'std': 0.0,
                'results': []
            },
            'memory_gpu': {
                'average': 0.0,
                'std': 0.0,
                'results': []
            },
            'time': {
                'average': 0.0,
                'std': 0.0,
                'results': []
            }
        }

        file_name = f"results_{model_properties['modelType']}_{model_properties['dataset']}_{model_properties['variant']}.json"
        folder = 'results'
        file_path = os.path.join(folder, file_name)

        if not os.path.exists(folder):
            os.makedirs(folder)

        # Check if the results have already been calculated for this model
        if os.path.exists(file_path):
            print(f"Results already exist for {file_name}. Loading results...")
            return self.load_results(file_path)


        print("Running execution time benchmark...")
        # Run the benchmark n_runs times for execution time only so that the memory profiler does not interfere with the results
        for i in range(n_runs):
            # Run the forward pass
            start_time = timeit.default_timer()
            model.forward(x_data)
            end_time = timeit.default_timer()

            # Record the time taken
            time = end_time - start_time
            results['time']['results'].append(time)

            print(f"Run {i+1} complete. Time taken: {time}")

        # Calculate the average and standard deviation of the time taken
        time_avg = sum(results['time']['results']) / n_runs
        time_std = (sum([(t - time_avg) ** 2 for t in results['time']['results']]) / n_runs) ** 0.5

        results['time']['average'] = time_avg
        results['time']['std'] = time_std

        print("Execution time benchmark complete.")
        print(f"Mean time taken: {time_avg}, Standard deviation: {time_std}")

        print("Running memory benchmark...")
        # Run the benchmark n_runs times for memory usage
        for i in range(n_runs):
            tracemalloc.start()
            # Reset the torch cuda memory tracker
            torch.cuda.reset_max_memory_allocated()
            torch.cuda.reset_max_memory_cached()

            model.forward(x_data)

            current, peak = tracemalloc.get_traced_memory()
            # Get the cuda statistics too
            cuda_current = torch.cuda.memory_allocated()
            cuda_peak = torch.cuda.max_memory_allocated()

            tracemalloc.stop()
            torch.cuda.reset_max_memory_allocated()
            torch.cuda.reset_max_memory_cached()

            # Record the memory usage
            results['memory_cpu']['results'].append((peak))
            results['memory_gpu']['results'].append((cuda_peak))

            print(f"Run {i+1} complete. \nMemory usage (MB): {current / 10**6}, Peak memory usage (MB): {peak / 10**6}\nCUDA memory usage (MB): {cuda_current / 10**6}, Peak CUDA memory usage (MB): {cuda_peak / 10**6}")

        # Calculate the average and standard deviation of the memory usage
        memory_avg = sum(results['memory_cpu']['results']) / n_runs
        memory_std = (sum([(m - memory_avg) ** 2 for m in results['memory_cpu']['results']]) / n_runs) ** 0.5

        cuda_memory_avg = sum(results['memory_gpu']['results']) / n_runs
        cuda_memory_std = (sum([(m - cuda_memory_avg) ** 2 for m in results['memory_gpu']['results']]) / n_runs) ** 0.5

        results['memory_cpu']['average'] = memory_avg
        results['memory_cpu']['std'] = memory_std

        results['memory_gpu']['average'] = cuda_memory_avg
        results['memory_gpu']['std'] = cuda_memory_std

        print("Memory benchmark complete.")

        print(f"Mean memory usage: {memory_avg}, Standard deviation: {memory_std}")
        print(f"Mean CUDA memory usage: {cuda_memory_avg}, Standard deviation: {cuda_memory_std}")

        self.save_results(results, file_path)

        return results

    def save_results(self, results, filename):
        # Save a given set of results as a JSON file
        with open(filename, 'w') as f:
            json.dump(results, f)

    def load_results(self, filename):
        # Load a set of results from a JSON file
        with open(filename) as f:
            return json.load(f)