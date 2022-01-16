from .aggregator_base import Aggregator
from collections import OrderedDict
import torch
import numpy as np
import copy

class FedAvg(Aggregator):
    def aggregate(self, local_models, global_model):
        if self.args.get_portion_uploaded() == 1:
            return super().aggregate(local_models, global_model)
        else:
            return self.fedavg_with_missing(local_models, global_model)

    def fedavg_with_missing(self, local_neurons, global_neurons):
        matched_local = [copy.deepcopy(i) for i in local_neurons]
        matched_copies = [copy.deepcopy(i) for i in matched_local]
        new_params = OrderedDict()

        for layer in global_neurons.keys():
            weights = [None for _ in matched_local]
            for i in range(len(matched_local)):
                matched_local[i][layer][torch.isnan(matched_local[i][layer])] = 0
                weights[i] = torch.ones(matched_local[i][layer].shape)
                weights[i][torch.isnan(matched_copies[i][layer])] = 0
            all_weights = np.array([j.cpu().numpy() for j in weights])
            weight_sum = np.sum(all_weights, axis=0)
            if len(weight_sum.shape) == 0:
                weight_sum = np.array([weight_sum])
            if 0 in weight_sum:
                all_weights[0][weight_sum == 0] = 1
                matched_local[0][layer].cpu().numpy()[weight_sum == 0] = global_neurons[layer].cpu().numpy()[
                    weight_sum == 0]
            new_params[layer] = torch.from_numpy(np.array(
                np.average(np.array([j[layer].cpu().numpy() for j in matched_local]), axis=0, weights=all_weights)))

        return new_params

    def simple_update_with_missing(self, client, new_params):
        self.args.get_logger().info("Updating parameters on client #{}", str(client.get_client_index()))
        local_params = client.get_nn_parameters()
        filled_update = OrderedDict()
        for layer in local_params.keys():
            #We replace NaNs with old params
            #print("Updated_params w NaNs:", new_params[layer])
            #print("Old_params:", local_params[layer])
            filled_update[layer] = torch.where(torch.isnan(new_params[layer].cpu()), local_params[layer].cpu(), new_params[layer].cpu())
            #print("Filled update:", filled_update[layer])
        client.update_nn_parameters(filled_update)

    def update_params(self, local_clients, global_client, new_params):
        if self.args.get_portion_downloaded() == 1:
            super().update_params(local_clients, global_client, new_params)
        else:
            self.args.get_logger().info("Updating global model parameters")
            global_client.update_nn_parameters(new_params)
            global_client.save_model(epoch=self.epoch, suffix='end')
            downloadable = self.get_downloadable_params()
            for client in local_clients:
                self.simple_update_with_missing(client, downloadable)
