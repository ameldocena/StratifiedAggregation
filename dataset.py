from abc import abstractmethod
from torch.utils.data import TensorDataset
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy

# Unmodified from original work
class Dataset:
    def __init__(self, args):
        self.args = args
        self.train_dataset = self.load_train_dataset()
        self.test_dataset = self.load_test_dataset()

    def get_args(self):
        """
        Returns the arguments.

        :return: Arguments
        """
        return self.args

    def get_train_dataset(self):
        """
        Returns the train dataset.

        :return: tuple
        """
        return self.train_dataset

    def get_test_dataset(self):
        """
        Returns the test dataset.

        :return: tuple
        """
        return self.test_dataset

    @abstractmethod
    def load_train_dataset(self):
        """
        Loads & returns the training dataset.

        :return: tuple
        """
        raise NotImplementedError("load_train_dataset() isn't implemented")

    @abstractmethod
    def load_test_dataset(self):
        """
        Loads & returns the test dataset.

        :return: tuple
        """
        raise NotImplementedError("load_test_dataset() isn't implemented")

    def get_train_loader(self, batch_size, **kwargs):
        """
        Return the data loader for the train dataset.

        :param batch_size: batch size of data loader
        :type batch_size: int
        :return: torch.utils.data.DataLoader
        """
        return Dataset.get_data_loader_from_data(batch_size, self.train_dataset[0], self.train_dataset[1], **kwargs)

    def get_test_loader(self, batch_size, **kwargs):
        """
        Return the data loader for the test dataset.

        :param batch_size: batch size of data loader
        :type batch_size: int
        :return: torch.utils.data.DataLoader
        """
        return Dataset.get_data_loader_from_data(batch_size, self.test_dataset[0], self.test_dataset[1], **kwargs)

    @staticmethod
    def get_data_loader_from_data(batch_size, X, Y, **kwargs):
        """
        Get a data loader created from a given set of data.

        :param batch_size: batch size of data loader
        :type batch_size: int
        :param X: data features
        :type X: numpy.Array()
        :param Y: data labels
        :type Y: numpy.Array()
        :return: torch.utils.data.DataLoader
        """
        X_torch = torch.from_numpy(X).float()

        if "classification_problem" in kwargs and kwargs["classification_problem"] == False:
            Y_torch = torch.from_numpy(Y).float()
        else:
            Y_torch = torch.from_numpy(Y).long()
        dataset = TensorDataset(X_torch, Y_torch)

        kwargs.pop("classification_problem", None)
        #A torch DataLoader object
        return DataLoader(dataset, batch_size=batch_size, **kwargs)

    @staticmethod
    def get_tuple_from_data_loader(data_loader):
        """
        Get a tuple representation of the data stored in a data loader.

        :param data_loader: data loader to get data from
        :type data_loader: torch.utils.data.DataLoader
        :return: tuple
        """
        return (next(iter(data_loader))[0].numpy(), next(iter(data_loader))[1].numpy())

class CIFAR10Dataset(Dataset):

    def __init__(self, args):
        super(CIFAR10Dataset, self).__init__(args)

    def load_train_dataset(self):
        self.get_args().get_logger().debug("Loading CIFAR10 train data")

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize
        ])
        train_dataset = datasets.CIFAR10(root=self.get_args().get_data_path(), train=True, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))

        train_data = self.get_tuple_from_data_loader(train_loader)

        self.get_args().get_logger().debug("Finished loading CIFAR10 train data")

        return train_data

    def load_test_dataset(self):
        self.get_args().get_logger().debug("Loading CIFAR10 test data")

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        print("Test data transforms:", transform)
        test_dataset = datasets.CIFAR10(root=self.get_args().get_data_path(), train=False, download=True, transform=transform)

        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

        test_data = self.get_tuple_from_data_loader(test_loader)

        self.get_args().get_logger().debug("Finished loading CIFAR10 test data")

        return test_data

class FashionMNISTDataset(Dataset):

    def __init__(self, args):
        super(FashionMNISTDataset, self).__init__(args)

    def load_train_dataset(self):
        self.get_args().get_logger().debug("Loading Fashion MNIST train data")

        train_dataset = datasets.FashionMNIST(self.get_args().get_data_path(), train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))

        train_data = self.get_tuple_from_data_loader(train_loader)

        self.get_args().get_logger().debug("Finished loading Fashion MNIST train data")

        return train_data

    def transpose(self, test_dataset):
        transposed = test_dataset.T
        return transposed

    def load_test_dataset(self):
        self.get_args().get_logger().debug("Loading Fashion MNIST test data")


        transform = transforms.ToTensor() #, transforms.RandomRotation((-90, 0))

        print("Test data transforms:", transform)
        test_dataset = datasets.FashionMNIST(self.get_args().get_data_path(), train=False, download=True, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

        test_data = self.get_tuple_from_data_loader(test_loader)

        self.get_args().get_logger().debug("Finished loading Fashion MNIST test data")

        return test_data
