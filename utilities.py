import random
import numpy
#import tensorflow as tf
#import torch
from abc import abstractmethod
from sklearn.decomposition import PCA
from aggregators import FedAvg, MultiKrum, AlignedAvg, TrimmedMean, Median, StratifiedAggr

class SelectionStrategy:
# Unchanged from original work
    @abstractmethod
    def select_round_workers(self, workers, poisoned_workers, kwargs):
        """
        :param workers: list(int). All workers available for learning
        :param poisoned_workers: list(int). All workers that are poisoned
        :param kwargs: dict
        """
        raise NotImplementedError("select_round_workers() not implemented")

class RandomSelectionStrategy(SelectionStrategy):
    # Unchanged from original work
    """
    Randomly selects workers out of the list of all workers
    """

    def select_round_workers(self, workers, poisoned_workers, kwargs):
        #The poisoned_workers here are not used
        return random.sample(workers, kwargs["NUM_WORKERS_PER_ROUND"])
        #returns a list of sampled worker ids

# class StratifiedRandomSelection(SelectionStrategy):
#     #We first stratify: Each stratum will be a list of workers
#     #Then within each stratum, we randomly select
#     #We would need the list of workers and the information about their skews

def select_aggregator(args, name, KWARGS={}):
    #Creates an Aggregator object as selected
    if name == "FedAvg":
        return FedAvg(args, name, KWARGS)
    elif name == "AlignedAvg":
        return AlignedAvg(args, name, KWARGS)
    elif name == "AlignedAvgImpute":
        KWARGS.update({"use_impute":"filter","align":"fusion"})
        return AlignedAvg(args, name, **KWARGS)
    elif name == "MultiKrum":
        return MultiKrum(args, name, KWARGS)
    elif name == "TrimmedMean":
        return TrimmedMean(args, name, KWARGS)
    elif name == "Median":
        return Median(args, name, KWARGS)
    elif (name == "StratKrum") or (name == "StratTrimMean") or (name == "StratMedian") or (name == "StratFedAvg"):
        #We may have to change the class name to StratifiedAggregation
        return StratifiedAggr(args, name, KWARGS)
    else:
        raise NotImplementedError(f"Unrecognized Aggregator Name: {name}")

def calculate_pca_of_gradients(logger, gradients, num_components):
    # Unchanged from original work
    pca = PCA(n_components=num_components)

    logger.info("Computing {}-component PCA of gradients".format(num_components))

    return pca.fit_transform(gradients)

#So this is here after all
def calculate_model_gradient( model_1, model_2):
    # Minor change from original work
    """
    Calculates the gradient (parameter difference) between two Torch models.

    :param logger: loguru.logger (NOW REMOVED)
    :param model_1: torch.nn
    :param model_2: torch.nn
    """
    model_1_parameters = list(dict(model_1.state_dict()))
    model_2_parameters = list(dict(model_2.state_dict()))

    return calculate_parameter_gradients(model_1_parameters, model_2_parameters)

def calculate_parameter_gradients(params_1, params_2):
    # Minor change from original work
    """
    Calculates the gradient (parameter difference) between two sets of Torch parameters.

    :param logger: loguru.logger (NOW REMOVED)
    :param params_1: dict
    :param params_2: dict
    """
    #logger.debug("Shape of model_1_parameters: {}".format(str(len(params_1))))
    #logger.debug("Shape of model_2_parameters: {}".format(str(len(params_2))))

    return numpy.array([x for x in numpy.subtract(params_1, params_2)])

# #Inserted
# def convert2TF(torch_tensor):
#     # Converts a pytorch tensor into a Tensorflow.
#     # We first convert torch into numpy, then to tensorflow.
#     # Arg: torch_tensor - a Pytorch tensor object
#     np_tensor = torch_tensor.numpy().astype(float)
#     return tf.convert_to_tensor(np_tensor)
#
# def convert2Torch(tf_tensor):
#     #Converts a TF tensor to Torch
#     #Arg: tf_tensor - a TF tensor
#     np_tensor = tf.make_ndarray(tf_tensor)
#     return torch.from_numpy(np_tensor)

def count_poisoned_stratum(stratified_workers, poisoned_workers):
    if len(poisoned_workers) > 0:
        print("\nPoisoned workers:", len(poisoned_workers), poisoned_workers)
        for stratum in stratified_workers:
            intersect = list(set(stratified_workers[stratum]).intersection(poisoned_workers))
            print("Count poisoned workers per stratum:", len(intersect), intersect)
            print("Stratum: {}. Propn to total poisoned: {}. Propn to subpopn in stratum: {}".format(stratum, len(intersect)/len(poisoned_workers),
                                                                                                 len(intersect)/len(stratified_workers[stratum])))
    else:
        print("No poisoned workers")


def generate_uniform_weights(random_workers):
    """
    This function generates uniform weights for each stratum in random_workers
    :param random_workers:
    :return:
    """
    strata_weights = dict()
    weight = 1.0 / len(list(random_workers.keys()))

    for stratum in random_workers:
        strata_weights[stratum] = weight

    return strata_weights