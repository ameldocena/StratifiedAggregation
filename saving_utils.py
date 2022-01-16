import csv

def convert_results_to_csv(results):
    """
    Unchanged from original work
    :param results: list(return data by test_classification() in client.py)
    """
    cleaned_epoch_test_set_results = []

    for row in results:
        components = [row[0], row[1]]

        for class_precision in row[2]:
            components.append(class_precision)
        for class_recall in row[3]:
            components.append(class_recall)

        cleaned_epoch_test_set_results.append(components)

    return cleaned_epoch_test_set_results

def generate_experiment_ids(start_idx, num_exp):
    # Unchanged from original work
    """
    Generate the filenames for all experiment IDs.

    :param start_idx: start index for experiments
    :type start_idx: int
    :param num_exp: number of experiments to run
    :type num_exp: int
    """
    log_files = []
    results_files = []
    models_folders = []
    worker_selections_files = []

    for i in range(num_exp):
        idx = str(start_idx + i)

        log_files.append("logs/" + idx + ".log")
        results_files.append(idx + "_results.csv")
        models_folders.append(idx + "_models")
        worker_selections_files.append(idx + "_workers_selected.csv")

    return log_files, results_files, models_folders, worker_selections_files

def save_results(results, filename):
    # Unchanged from original work
    """
    :param results: experiment results
    :type results: list()
    :param filename: File name to write results to
    :type filename: String
    """
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        for experiment in results:
            writer.writerow(experiment)

def get_class_label_from_num(dataset, class_num):
    # Function I added
    if dataset == 'FashionMNIST':
        mapping = {0:"T-shirt/top",
                   1:"trouser",
                   2:"pullover",
                   3:"dress",
                   4:"coat",
                   5:"sandal",
                   6:"shirt",
                   7:"sneaker",
                   8:"bag",
                   9:"ankle boot"}
    elif dataset == 'CIFAR10':
        mapping = {0: "airplane",
                   1: "automobile",
                   2: "bird",
                   3: "cat",
                   4: "deer",
                   5: "dog",
                   6: "frog",
                   7: "horse",
                   8: "ship",
                   9: "truck"}
    else:
        raise NotImplementedError(f"Unrecognized Dataset Name: {dataset}")
    return mapping[class_num]


