import os
import pickle
import numpy as np
import random
from dataset import Dataset

def load_data_loader(logger, args, test=True):
    # Unchanged from original work
    if test:
        if os.path.exists(args.get_test_data_loader_pickle_path()):
            logger.info("Loading data loader from file: {}".format(args.get_test_data_loader_pickle_path()))
            with open(args.get_test_data_loader_pickle_path(), "rb") as f:
                return pickle.load(f)
        else:
            logger.error("Couldn't find test data loader stored in file")
            raise FileNotFoundError("Couldn't find train data loader stored in file")

    else:
        if os.path.exists(args.get_train_data_loader_pickle_path()):
            logger.info("Loading data loader from file: {}".format(args.get_train_data_loader_pickle_path()))
            with open(args.get_train_data_loader_pickle_path(), "rb") as f:
                return pickle.load(f)
        else:
            logger.error("Couldn't find train data loader stored in file")
            raise FileNotFoundError("Couldn't find train data loader stored in file")

def generate_data_loader(args, dataset, test=True):
    # Unchanged from original work
    if test:
        t_dataset = dataset.get_test_dataset()
    else:
        t_dataset = dataset.get_train_dataset()
    X, Y = shuffle_data(t_dataset)

    return dataset.get_data_loader_from_data(args.get_batch_size(), X, Y)

def generate_data_loaders_from_distributed_dataset(distributed_dataset, batch_size):
    # Unchanged from original work
    """
    Generate data loaders from a distributed dataset.
    A distributed dataset is a dict with X and Y for each worker ID

    :param distributed_dataset: Distributed dataset
    :type distributed_dataset: list(tuple)
    :param batch_size: batch size for data loader
    :type batch_size: int
    """
    data_loaders = []
    for worker_training_data in distributed_dataset:
        "This is the part where the bug is. It says that the data_loaders are missing"
        data_loaders.append(Dataset.get_data_loader_from_data(batch_size, worker_training_data[0], worker_training_data[1], shuffle=True))

    return data_loaders

def shuffle_data(dataset):
    # Unchanged from original work
    data = list(zip(dataset[0], dataset[1]))
    random.shuffle(data)
    X, Y = zip(*data)
    X = np.asarray(X)
    Y = np.asarray(Y)

    return X, Y