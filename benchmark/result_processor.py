import json
import os.path

if __name__ == '__main__':
    results_file = 'results.json'

    results = []

    if os.path.exists(results_file):
        # Load the results from the file
        with open(results_file, 'r') as f:
            results = json.load(f)

    if len(results) == 0:
        print("No results found.")
        # Exit the program
        exit()

    # Print the results
    for result in results:
        print(f"Model Type: {result['modelType']}, Dataset: {result['dataset']}, Variant: {result['variant']}")
        print("============== Timing Results ==============")
        print(f"Mean: {result['time']['average']}s")
        print(f"Standard Deviation: {result['time']['std']}s")

        print("============== Host Memory Usage ==============")
        # The memory usage is in Bytes, convert to MB
        print(f"Mean: {result['memory_cpu']['average'] / 1024 / 1024}MB")
        print(f"Standard Deviation: {result['memory_cpu']['std'] / 1024 / 1024}MB")

        print("============== GPU Memory Usage ==============")
        # The memory usage is in Bytes, convert to MB
        print(f"Mean: {result['memory_gpu']['average'] / 1024 / 1024}MB")
        print(f"Standard Deviation: {result['memory_gpu']['std'] / 1024 / 1024}MB")

        print("==============================================\n\n")

    # Construct a table of the results, in the format:
    # Model | Mean Time | Std Time | Mean CPU Memory | Std CPU Memory | Mean GPU Memory | Std GPU Memory
    print("Model\t|Mean Time\t|Std Time\t|Mean CPU Memory\t|Std CPU Memory\t|Mean GPU Memory\t|Std GPU Memory")
    for result in results:
        print(f"{result['modelType']} {result['variant']}\t|{result['time']['average']}s\t|{result['time']['std']}s\t|{result['memory_cpu']['average'] / 1024 / 1024}MB\t|{result['memory_cpu']['std'] / 1024 / 1024}MB\t|{result['memory_gpu']['average'] / 1024 / 1024}MB\t|{result['memory_gpu']['std'] / 1024 / 1024}MB")

    # Export a CSV file of the results
    with open('results.csv', 'w') as f:
        f.write("Model,Variant,Mean Time,Std Time,Mean CPU Memory,Std CPU Memory,Mean GPU Memory,Std GPU Memory\n")
        for result in results:
            f.write(f"{result['modelType']},{result['variant']},{result['time']['average']},{result['time']['std']},{result['memory_cpu']['average'] / 1024 / 1024},{result['memory_cpu']['std'] / 1024 / 1024},{result['memory_gpu']['average'] / 1024 / 1024},{result['memory_gpu']['std'] / 1024 / 1024}\n")

    print("Results exported to results.csv")
    print("Exiting...")
