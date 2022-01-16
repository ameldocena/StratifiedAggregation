from nets import Cifar10CNN
from nets import FashionMNISTCNN
import torch
import json

# Setting the seed for Torch
SEED = 1
torch.manual_seed(SEED)

class Arguments:
    # Minor changes from original work
    def __init__(self, logger):
        self.logger = logger
        # Commented values should be swapped when changing from CIFAR-10 to Fashion-MNIST, also self.net and data loader paths
        self.batch_size = 10
        self.test_batch_size = 1000
        self.epochs = 100 # default 10
        self.lr = 0.001 # 0.01 #  default 0.1
        self.momentum = 0.9 # 0.5 # default 0.5
        self.cuda = True # default True
        self.shuffle = False
        self.log_interval = 10 #100
        self.kwargs = {}

        self.scheduler_step_size = 10 # 50 #  default 50
        self.scheduler_gamma = 0.1 # 0.5 # default 0.5
        self.min_lr = 1e-10

        self.round_worker_selection_strategy = None
        self.round_worker_selection_strategy_kwargs = None

        self.save_model = True # Default False, modified
        self.save_epoch_interval = 1
        self.save_model_path = "models"
        self.epoch_save_start_suffix = "start"
        self.epoch_save_end_suffix = "end"
        self.randomize_start = True

        self.num_workers = 100
        self.num_poisoned_workers = 0 # Default 0

        #self.net = Cifar10CNN
        self.net = FashionMNISTCNN

        self.train_data_loader_pickle_path = "data_loaders/fashion-mnist/train_data_loader.pickle"
        self.test_data_loader_pickle_path = "data_loaders/fashion-mnist/test_data_loader.pickle"

        #self.train_data_loader_pickle_path = "data_loaders/cifar10/train_data_loader.pickle"
        #self.test_data_loader_pickle_path = "data_loaders/cifar10/test_data_loader.pickle"

        self.loss_function = torch.nn.CrossEntropyLoss

        self.default_model_folder_path = "default_models"

        self.data_path = "data"

        self.portion_downloaded = 1

        self.portion_uploaded = 1

        self.aggregator = None

    def switch_CIFAR_MNIST(self, dataset):
        if dataset == "CIFAR10":
            self.batch_size = 10
            self.lr = 0.01
            self.momentum = 0.0 #0.5
            self.scheduler_step_size = 50
            self.scheduler_gamma = 0.5
            self.net = Cifar10CNN
            self.train_data_loader_pickle_path = "data_loaders/cifar10/train_data_loader.pickle"
            self.test_data_loader_pickle_path = "data_loaders/cifar10/test_data_loader.pickle"
        elif dataset == "FashionMNIST":
            self.batch_size = 4
            self.lr = 0.001
            self.momentum = 0.0 #0.9
            self.scheduler_step_size = 10
            self.scheduler_gamma = 0.1
            self.net = FashionMNISTCNN
            self.train_data_loader_pickle_path = "data_loaders/fashion-mnist/train_data_loader.pickle"
            self.test_data_loader_pickle_path = "data_loaders/fashion-mnist/test_data_loader.pickle"
        else:
            raise NotImplementedError(f"Unrecognized Dataset: {dataset}")

    def set_aggregator(self, aggregator):
        self.aggregator = aggregator

    def set_randomized_start(self, randomize):
        self.randomize_start = randomize

    def set_portion_downloaded(self, portion):
        self.portion_downloaded = portion

    def get_portion_downloaded(self):
        return self.portion_downloaded

    def set_portion_uploaded(self, portion):
        self.portion_uploaded = portion

    def get_portion_uploaded(self):
        return self.portion_uploaded

    def get_round_worker_selection_strategy(self):
        return self.round_worker_selection_strategy

    def get_round_worker_selection_strategy_kwargs(self):
        return self.round_worker_selection_strategy_kwargs

    def set_round_worker_selection_strategy_kwargs(self, kwargs):
        self.round_worker_selection_strategy_kwargs = kwargs

    def set_client_selection_strategy(self, strategy):
        self.round_worker_selection_strategy = strategy

    def get_data_path(self):
        return self.data_path

    def get_epoch_save_start_suffix(self):
        return self.epoch_save_start_suffix

    def get_epoch_save_end_suffix(self):
        return self.epoch_save_end_suffix

    def set_train_data_loader_pickle_path(self, path):
        self.train_data_loader_pickle_path = path

    def get_train_data_loader_pickle_path(self):
        return self.train_data_loader_pickle_path

    def set_test_data_loader_pickle_path(self, path):
        self.test_data_loader_pickle_path = path

    def get_test_data_loader_pickle_path(self):
        return self.test_data_loader_pickle_path

    def get_cuda(self):
        return self.cuda

    def get_scheduler_step_size(self):
        return self.scheduler_step_size

    def get_scheduler_gamma(self):
        return self.scheduler_gamma

    def get_min_lr(self):
        return self.min_lr

    def get_default_model_folder_path(self):
        return self.default_model_folder_path

    def get_num_epochs(self):
        return self.epochs

    def set_num_poisoned_workers(self, num_poisoned_workers):
        self.num_poisoned_workers = num_poisoned_workers

    def set_num_workers(self, num_workers):
        self.num_workers = num_workers

    def set_num_selected_workers(self, num_selected_workers):
        self.num_selected_workers = num_selected_workers

    def set_model_save_path(self, save_model_path):
        self.save_model_path = save_model_path

    def get_logger(self):
        return self.logger

    def get_loss_function(self):
        return self.loss_function

    def get_net(self):
        return self.net

    def get_num_workers(self):
        return self.num_workers

    def get_num_poisoned_workers(self):
        return self.num_poisoned_workers

    def get_learning_rate(self):
        return self.lr

    def get_momentum(self):
        return self.momentum

    def get_shuffle(self):
        return self.shuffle

    def get_batch_size(self):
        return self.batch_size

    def get_test_batch_size(self):
        return self.test_batch_size

    def get_log_interval(self):
        return self.log_interval

    def get_save_model_folder_path(self):
        return self.save_model_path

    def get_learning_rate_from_epoch(self, epoch_idx):
        lr = self.lr * (self.scheduler_gamma ** int(epoch_idx / self.scheduler_step_size))

        if lr < self.min_lr:
            self.logger.warning("Updating LR would place it below min LR. Skipping LR update.")

            return self.min_lr

        self.logger.debug("LR: {}".format(lr))

        return lr

    def should_save_model(self, epoch_idx):
        """
        Returns true/false models should be saved.

        :param epoch_idx: current training epoch index
        :type epoch_idx: int
        """
        if not self.save_model:
            return False

        if epoch_idx == 1 or epoch_idx % self.save_epoch_interval == 0:
            return True

    def log(self):
        """
        Log this arguments object to the logger.
        """
        self.logger.debug("Arguments: {}", str(self))

    def __str__(self):
        return "\nBatch Size: {}\n".format(self.batch_size) + \
               "Test Batch Size: {}\n".format(self.test_batch_size) + \
               "Epochs: {}\n".format(self.epochs) + \
               "Learning Rate: {}\n".format(self.lr) + \
               "Momentum: {}\n".format(self.momentum) + \
               "CUDA Enabled: {}\n".format(self.cuda) + \
               "Shuffle Enabled: {}\n".format(self.shuffle) + \
               "Log Interval: {}\n".format(self.log_interval) + \
               "Scheduler Step Size: {}\n".format(self.scheduler_step_size) + \
               "Scheduler Gamma: {}\n".format(self.scheduler_gamma) + \
               "Scheduler Minimum Learning Rate: {}\n".format(self.min_lr) + \
               "Client Selection Strategy: {}\n".format(self.round_worker_selection_strategy) + \
               "Client Selection Strategy Arguments: {}\n".format(json.dumps(self.round_worker_selection_strategy_kwargs, indent=4, sort_keys=True)) + \
               "Model Saving Enabled: {}\n".format(self.save_model) + \
               "Model Saving Interval: {}\n".format(self.save_epoch_interval) + \
               "Model Saving Path (Relative): {}\n".format(self.save_model_path) + \
               "Epoch Save Start Prefix: {}\n".format(self.epoch_save_start_suffix) + \
               "Epoch Save End Suffix: {}\n".format(self.epoch_save_end_suffix) + \
               "Number of Clients: {}\n".format(self.num_workers) + \
               "Number of Poisoned Clients: {}\n".format(self.num_poisoned_workers) + \
               "NN: {}\n".format(self.net) + \
               "Train Data Loader Path: {}\n".format(self.train_data_loader_pickle_path) + \
               "Test Data Loader Path: {}\n".format(self.test_data_loader_pickle_path) + \
               "Loss Function: {}\n".format(self.loss_function) + \
               "Default Model Folder Path: {}\n".format(self.default_model_folder_path) + \
               "Data Path: {}\n".format(self.data_path) + \
               "Parameters uploaded to server: {}\n".format(self.portion_uploaded) + \
               "Parameters downloaded to workers: {}\n".format(self.portion_downloaded) + \
               "Aggregation algorithm: {}\n".format(str(self.aggregator))
