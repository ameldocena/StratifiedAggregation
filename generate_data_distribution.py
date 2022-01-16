from loguru import logger
import pathlib
import os
from arguments import Arguments
from dataset import CIFAR10Dataset
from dataset import FashionMNISTDataset
from dataloaders import generate_data_loader
import pickle
# Unmodified from original work

if __name__ == '__main__':
    args = Arguments(logger)

    # ---------------------------------
    # ------------ CIFAR10 ------------
    # ---------------------------------
    dataset = CIFAR10Dataset(args)
    TRAIN_DATA_LOADER_FILE_PATH = "data_loaders/cifar10/train_data_loader.pickle"
    TEST_DATA_LOADER_FILE_PATH = "data_loaders/cifar10/test_data_loader.pickle"

    if not os.path.exists("data_loaders/cifar10"):
        pathlib.Path("data_loaders/cifar10").mkdir(parents=True, exist_ok=True)

    train_data_loader = generate_data_loader(args, dataset, test=False)
    test_data_loader = generate_data_loader(args, dataset, test=True)

    with open(TRAIN_DATA_LOADER_FILE_PATH, "wb") as f:
        pickle.dump(train_data_loader, f)

    with open(TEST_DATA_LOADER_FILE_PATH, "wb") as f:
        pickle.dump(test_data_loader, f)

    # ---------------------------------
    # --------- Fashion-MNIST ---------
    # ---------------------------------
    dataset = FashionMNISTDataset(args)
    TRAIN_DATA_LOADER_FILE_PATH = "data_loaders/fashion-mnist/train_data_loader.pickle"
    TEST_DATA_LOADER_FILE_PATH = "data_loaders/fashion-mnist/test_data_loader.pickle"

    if not os.path.exists("data_loaders/fashion-mnist"):
        pathlib.Path("data_loaders/fashion-mnist").mkdir(parents=True, exist_ok=True)

    train_data_loader = generate_data_loader(args, dataset, test=False)
    test_data_loader = generate_data_loader(args, dataset, test=True)

    with open(TRAIN_DATA_LOADER_FILE_PATH, "wb") as f:
        pickle.dump(train_data_loader, f)

    with open(TEST_DATA_LOADER_FILE_PATH, "wb") as f:
        pickle.dump(test_data_loader, f)
