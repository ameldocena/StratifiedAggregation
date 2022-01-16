import numpy as np
import torch
from collections import OrderedDict
from .aggregator_base import Aggregator
from .fedavg import FedAvg


class MultiKrum(Aggregator):
  """ Full-TensorFlow Multi-Krum GAR class with restricted gradient sharing for upload and restricted parameter sharing
  for download.
  """
  def __init__(self, arguments, name, kwargs):
    print("class MultiKrum is used")
    super().__init__(arguments, name)
    """
    Here, we set the number of workers per round and our assumed number of byzantine workers selected during that round
    """
    if name=="MultiKrum":
      self.__nbworkers  = kwargs['NUM_WORKERS_PER_ROUND'] #notated as n
      self.__nbbyzwrks  = kwargs["ASSUMED_POISONED_WORKERS_PER_ROUND"] #notated as f

    else: #StratKrum
      self.__nbworkers = kwargs['assumed_workers_per_round_stratum']  # notated as n
      self.__nbbyzwrks = kwargs["assumed_poisoned_per_round_stratum"]  # notated as f

    self.__nbselected = self.__nbworkers - self.__nbbyzwrks - 2
    self.k = kwargs['NUM_KRUM']
    self.portion_uploaded = self.args.get_portion_uploaded()
    self.portion_downloaded = self.args.get_portion_downloaded()
    print("Krum workers:", self.__nbworkers, self.__nbbyzwrks, self.__nbselected)

    #Some assertions for restricted portions
    assert self.portion_uploaded > 0 and self.portion_uploaded <= 1, "Upload proportion should be within (0, 1]"
    assert self.portion_downloaded > 0 and self.portion_downloaded <= 1, "Download proportion should be within (0, 1]"

  def restricted_grad_sharing(self, gradients, portion):
    """
    Restricts sharing of gradients according to the portion stipulated. For those gradients restricted, these are replaced
    with 0 values.

    Inputs:
    gradients - gradients, (stored as OrderedDict)
    portion - portion to be shared, (scalar)

    Return:
    shared_grads - gradients but with some portion restricted for sharing replaced with 0 values
    """
    print("Restricted gradient sharing:", portion)
    if portion == 1:
      return gradients
    else:
      shared_grads = OrderedDict()
      for layer in gradients.keys():
        gradient = gradients[layer]
        null = torch.zeros(gradient.shape, dtype = gradient.dtype) #Container of zeros to replace restricted parameters
        mask = torch.full_like(gradient, False) #Initialize mask with False values
        # Sort the gradient in descending order, then slice up to the unrestricted portion; these indices shall take gradient values.
        mask.view(-1)[torch.sort(torch.abs(gradient).view(-1), descending=True).indices[:round(mask.numel() * portion)]] = True
        # If mask is True, set corresponding gradient value, else fill with 0
        shared_grads[layer] = torch.where(mask == True, gradient, null)
        #print("Orig grad: {}. Restricted grad: {}".format(gradients[layer], shared_grads[layer]))
      return shared_grads

  def get_gradient(self, idx, epoch, portion):
    #idx - worker index
    #epoch - communication round
    # This function collects the uploaded gradients of clients for a given epoch, and converts these from Torch to numpy
    folder = self.args.get_save_model_folder_path()
    end_path = f"{folder}/model_{idx}_{epoch}_end.model"
    end_model = torch.load(end_path)

    start_path = f"{folder}/model_{idx}_{epoch}_start.model"
    start_model = torch.load(start_path)
    #print("Start model:", type(start_model))
    #print("End model:", type(end_model))
    #The start and end models are ordered dicts for each of the parameters per layer

    grad_dict = OrderedDict()

    #Recall the gradient per layer and convert to numpy float
    #print("Start_model keys: {}. Length: {}".format(start_model.keys(), len(list(start_model.keys()))))
    #print("Running mean old: {}. New: {}".format(start_model['bn1.running_mean'], end_model['bn1.running_mean']))
    for i in start_model.keys():
      grad_dict[i] = start_model[i] - end_model[i] #That is: new_mod = old_mod - lr*grad => lr*grad = old_mod - new_mod

    restricted_grad_dict = self.restricted_grad_sharing(grad_dict, portion)
    # We can turn this into OrderedDict() similar to that of
    #print("Length of grad_dict", len(list(grad_dict.values())))
    #print("Gradient dict:", grad_dict.keys())

    #print("Length manual gradient:", len(list(grad_dict.values())))
    #gradient = calculate_model_gradient(end_model, start_model)
    #print("Coded gradient:", gradient.shape)

    #Convert Torch to numpy
    for i in start_model.keys():
      restricted_grad_dict[i] = restricted_grad_dict[i].numpy()

    return list(restricted_grad_dict.values())  # list of gradients per layer as numpy float
    #return gradient
    #If in aggregation we use ordered dict as well, then we use ordered dict; otherwise, we use list of gradients.

  def get_uploaded_gradients(self):
    """
    Collects the gradients of each worker given the epoch.

    Returns: Worker and its gradients (list)
    """

    #print("Epoch: {}. Current workers: {}".format(self.epoch, self.current_round))
    #return [self.get_gradient(idx, self.epoch) for idx in self.current_round]

    worker_grad_dict = dict()
    for idx in self.current_round:
      worker_grad_dict[idx] = self.get_gradient(idx, self.epoch, self.portion_uploaded)

    return worker_grad_dict

  #I should use local models and global model as arguments to the aggregate function
  #Depends actually on what we are trying to run here.
  def aggregate(self, uploaded_gradients):
    "Computes the Multi-Krum gradient"

    #Uploaded_gradients is a dictionary of (worker: gradient) key-value pair.
    #Note that the 'gradient' value is a list of gradients per layer.

    # Non-empty gradients assertion
    assert len(uploaded_gradients.values()) > 0, "Empty list of gradient to aggregate"
    num_layers = len(uploaded_gradients[list(uploaded_gradients.keys())[0]]) #number of layers

    # Distance computations from worker i to the rest of workers, indexed j
    dist_bet_workers = dict()
    scores_workers = dict()
    ## Computing the scores: the distance among the closest neighbors
    selected_workers = uploaded_gradients.keys() #or we can use self.current_road
    for i in selected_workers:
      dist_bet_workers[i] = list() #container for distances between i and other j workers
      for j in selected_workers:
        sqr_dist_layers = list()
        if i != j:
          for layer in range(num_layers):
            #Measure the sum-squared-distance of gradients per layer between worker i and j
            sqr_dist_per_layer = np.sum(np.square(uploaded_gradients[i][layer] - uploaded_gradients[j][layer]))
            sqr_dist_layers.append(sqr_dist_per_layer)
          #We sum up all the squared distance per layer to get the total distance between their gradients
          dist_bet_workers[i].append(np.sum(np.array(sqr_dist_layers)))
      #To get the score per worker, we sum up the distances of the nearest n-f-2 workers
      #a. Sort dist_bet_workers['i'] increasing order
      dist_bet_workers[i].sort()
      #Slice upto [:n-f-2] and sum up these distances to get the score
      scores_workers[i] = np.sum(dist_bet_workers[i][:self.__nbselected])

    #We now compute the Krum gradient
    #If k = 1, then this merely returns the gradient of the worker who has the minimum score.
    #If k > 1, then the average of the gradients of the bottom-k workers in terms of score.
    #Whatever the value of k is, the average operation works/is valid.

    #Sort scores and pick the bottom-k workers
    print("Scores:", scores_workers)
    candidates = sorted(scores_workers.keys(), key = lambda x: scores_workers[x])[:self.k] #A list of string candidate-worker ids
    krum_gradient = list()
    print("Candidates:", candidates)
    for layer in range(num_layers):
      sum_layer_gradients = uploaded_gradients[candidates[0]][layer]
      for i in range(1, len(candidates)):
        sum_layer_gradients += uploaded_gradients[candidates[i]][layer]
      krum_gradient.append(sum_layer_gradients/len(candidates))
    #print("Diff between min Candidate {} and Krum: {}".format(candidates[0], np.sum(uploaded_gradients[candidates[0]][0] - krum_gradient[0])))

    #I can either return in numpy or replace as torch tensor
    print("Return: Krum gradient")
    return krum_gradient #A list of gradients stored as numpy per layer

  """
  To update parameters:
  1. Update global parameters first.
  2. Using the updated global parameters, send these for download (update) to local clients according to restriction.
  """

  def update_global_params(self, global_client, agg_gradient, save = True):
    # This will update the global parameters using the aggregated gradient, so as the local clients' parameters
    # Note that the agg_gradient used here already takes into account restriction of uploaded gradients

    global_parameters = global_client.get_nn_parameters()  # A dictionary; make sure what are the contents
    updated_params = global_parameters.copy()
    layers = list(global_parameters.keys())  # layers of the global model
    # print("Global params: {}. Length: {}".format(layers, len(list(layers))))
    # print("bn6.num_batches_tracked:", global_parameters['bn6.running_var'])
    for i in range(len(layers)):
      updated_params[layers[i]] = global_parameters[layers[i]] - agg_gradient[i]

    global_client.update_nn_parameters(updated_params)
    if save is True:
      self.args.get_logger().info("Updating global model parameters")
      global_client.save_model(epoch=self.epoch, suffix='end')  # Saved as Global end model given the epoch
    return updated_params

  def update_params(self, local_clients, global_client, agg_gradient):
    # A. Update global parameters
    updated_global_params = self.update_global_params(global_client, agg_gradient)

    # B. Update local parameters
    print("Portion downloaded by clients:", self.portion_downloaded)
    for client in local_clients:
      ##If download = 1.0, update all local clients with full updated_params
      ##Else, use FedAvg.simple_update_with_missing to account for restricted download of paramaters
      if self.portion_downloaded == 1:
        self.args.get_logger().info("Unrestricted download of global parameters on client #{}", str(client.get_client_index()))
        client.update_nn_parameters(updated_global_params)

      else:
        downloadable = super().get_downloadable_params() #Restricted global parameters for download
        FedAvg.simple_update_with_missing(self, client, downloadable)

  #This is how this will be used: run_aggregation(epoch, random_workers, clients, global_model)
  def run_aggregation(self, epoch, client_ids, local_clients, global_client):
      #Objective: This should update the global and downloadable parameters
      self.epoch = epoch
      self.current_round = client_ids

      uploaded_gradients = self.get_uploaded_gradients()
      #print("Uploaded gradients type:", type(uploaded_gradients), len(uploaded_gradients))
      #print("Sample gradient:", type(uploaded_gradients[0]), len(uploaded_gradients[0]))

      self.args.get_logger().info("Aggregating client gradients using {}", str(self))
      krum_gradient = self.aggregate(uploaded_gradients)

      self.update_params(local_clients, global_client, krum_gradient)