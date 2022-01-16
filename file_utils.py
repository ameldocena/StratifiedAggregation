
def get_poisoned_worker_ids_from_log(log_path):
    # Unchanged from original work
    """
    :param log_path: string
    """
    with open(log_path, "r") as f:
        file_lines = [line.strip() for line in f.readlines() if "Poisoning data for workers:" in line]

    workers = file_lines[0].split("[")[1].split("]")[0]
    workers_list = workers.split(", ")

    return [int(worker) for worker in workers_list]

def get_worker_num_from_model_file_name(model_file_name):
    # Unchanged from original work
    """
    :param model_file_name: string
    """
    return int(model_file_name.split("_")[1])

def get_epoch_num_from_model_file_name(model_file_name):
    # Unchanged from original work
    """
    :param model_file_name: string
    """
    return int(model_file_name.split("_")[2].split(".")[0])

def get_suffix_from_model_file_name(model_file_name):
    # Unchanged from original work
    """
    :param model_file_name: string
    """
    return model_file_name.split("_")[3].split(".")[0]

def get_model_files_for_worker(model_files, worker_id):
    # Unchanged from original work
    """
    :param model_files: list[string]
    :param worker_id: int
    """
    worker_model_files = []

    for model in model_files:
        worker_num = get_worker_num_from_model_file_name(model)

        if worker_num == worker_id:
            worker_model_files.append(model)

    return worker_model_files

def get_model_files_for_epoch(model_files, epoch_num):
    # Unchanged from original work
    """
    :param model_files: list[string]
    :param epoch_num: int
    """
    epoch_model_files = []

    for model in model_files:
        model_epoch_num = get_epoch_num_from_model_file_name(model)

        if model_epoch_num == epoch_num:
            epoch_model_files.append(model)

    return epoch_model_files

def get_model_files_for_suffix(model_files, suffix):
    """
    :param model_files: list[string]
    :param suffix: string
    """
    suffix_only_model_files = []

    for model in model_files:
        model_suffix = get_suffix_from_model_file_name(model)

        if model_suffix == suffix:
            suffix_only_model_files.append(model)

    return suffix_only_model_files