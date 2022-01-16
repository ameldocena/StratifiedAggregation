# aligned_average, aligned_average_with_missing, aligned_average_impute

from .fedavg import FedAvg
from collections import OrderedDict
import copy
import torch
from sklearn.metrics import pairwise
from scipy.optimize import linear_sum_assignment
from scipy.signal import correlate
import re
import ot
import numpy as np

class AlignedAvg(FedAvg):
    def __init__(self, args, name, use_impute=False, align=None, **kwargs):
        super(AlignedAvg, self).__init__(args, name)
        self.use_impute = use_impute
        self.align = align
        self.__dict__.update(kwargs)

    def impute(self, model, func=lambda x: torch.where(torch.isnan(x), torch.zeros(x.shape, device=x.device, dtype=x.dtype), x)):
        imputed = OrderedDict()
        for layer in model.keys():
            imputed[layer] = func(copy.deepcopy(model[layer]))
        return imputed

    @staticmethod
    def filter_wise_mean(layer):
        """
        Impute missing convolutional weight values using the filter-wise mean where available
        For bias values, use layer-wise mean where available
        For FC weights, use average across largest tensor dimension (e.g., [[0,1,NaN],[0,NaN,2]] -> [[0,1,0.5],[0,1,2]])
        :param layer:
        :return:
        TODO: See if this is 2d or 3d filter
        TODO: Add channel-wise mean
        """
        if len(layer.shape) == 0:
            # Batch norm num batches tracked
            if np.isnan(layer.item()):
                return torch.zeros(layer.shape, device=layer.device, dtype=layer.dtype)
            else:
                return layer
        elif len(layer.shape) == 1:
            # Bias and other batch norm parameters
            return AlignedAvg.layer_wise_mean(layer)
        elif len(layer.shape) == 4:
            # Convolutional weights
            fill = copy.deepcopy(torch.from_numpy(np.nanmean(layer, axis=(2,3), keepdims=True)).expand(layer.shape))
            return torch.where(torch.isnan(layer), fill, layer)
        elif len(layer.shape) == 2:
            # FC layer weights
            dim = np.argmax(np.array([layer.shape[0], layer.shape[1]]))
            fill = copy.deepcopy(torch.from_numpy(np.nanmean(layer, axis=dim, keepdims=True)).expand(layer.shape))
            return torch.where(torch.isnan(layer), fill, layer)
        else:
            raise NotImplementedError(f"Unexpected tensor shape during imputation: {layer.shape}")

    @staticmethod
    def layer_wise_mean(layer):
        if np.isnan(np.nanmean(layer)):
            return torch.zeros(layer.shape, device=layer.device, dtype=layer.dtype)
        layer[torch.isnan(layer)] = torch.from_numpy(np.array(np.nanmean(layer)), device=layer.device, dtype=layer.dtype)
        return layer

    def aggregate(self, local_clients, local_models, global_model):
        if self.args.get_portion_uploaded() < 1.0:
            if self.use_impute == "filter":
                local_models = [self.impute(model, AlignedAvg.filter_wise_mean) for model in local_models]
            elif self.use_impute == "layer":
                local_models = [self.impute(model, AlignedAvg.layer_wise_mean) for model in local_models]
            elif self.use_impute == False:
                local_models = [self.impute(model) for model in local_models]
            else:
                raise NotImplementedError(f"Unrecognized Imputation Strategy: {self.use_impute}")
        upload_ix = 0
        local_aligned = []
        for idx in self.current_round:
            assert local_clients[idx].get_client_index() == idx, "Client ID Error"
            aligned_model = self.get_alignment(local_clients[idx], local_models[upload_ix], global_model)
            local_aligned.append(aligned_model)
            upload_ix+=1
        new_params = self.fedavg_with_missing(local_aligned, global_model)
        return new_params


    @staticmethod
    def _align_matrix(local_filter, global_filter, **kwargs):
        shape = local_filter.shape
        if 'distance' not in kwargs.keys() or kwargs['distance'] == 'cosine':
            cost = pairwise.cosine_distances(local_filter, global_filter)
        elif kwargs['distance'] == 'manhattan':
            cost = pairwise.manhattan_distances(local_filter, global_filter)
        elif kwargs['distance'] == 'euclidean':
            cost = pairwise.euclidean_distances(local_filter, global_filter)
        # elif kwargs['distance'] == 'correlation':
        #    correlate()
        else:
            raise NotImplementedError(f"Unrecognized distance metric: {kwargs['distance']}")
        if 'solver' not in kwargs.keys() or kwargs['solver'] == 'hungarian':
            alignment = linear_sum_assignment(cost / cost.max())
            alignment = alignment[1]
        elif kwargs['solver'] == 'emd':
            alignment = ot.emd([1 / shape[0] for _ in range(shape[0])], [1 / shape[0] for _ in range(shape[0])], cost / cost.max())
            alignment = alignment.argmax(alignment, axis=1)
        else:
            raise NotImplementedError(f"Unrecognized optimizer: {kwargs['optimizer']}")
        return alignment

    @staticmethod
    def _get_previous_layer(state_dict, layer):
        prev = None
        for i in state_dict.keys():
            if i == layer:
                return prev
            elif 'num_batches_tracked' not in i:
                prev = i
            else:
                continue

    @staticmethod
    def _update_alignment(client, alignments):
        aligned_model = OrderedDict()
        unaligned_model = client.get_nn_parameters()
        last_layer_weight = list(unaligned_model.keys())[-2]
        last_layer_bias = list(unaligned_model.keys())[-1]
        first_layer_weight = list(unaligned_model.keys())[0]
        for layer in client.state_dict().keys():
            if np.all(alignments[layer] == np.array([i for i in range(len(alignments[layer]))])):
                aligned_model[layer] = copy.deepcopy(unaligned_model[layer])
            elif len(unaligned_model[layer].shape) == 0:
                # Batch Norm Num Batches Tracked
                aligned_model[layer] = copy.deepcopy(unaligned_model[layer])
            elif layer == last_layer_bias:
                # Last FC Layer Bias is never re-aligned
                aligned_model[layer] = copy.deepcopy(unaligned_model[layer])
            elif len(unaligned_model[layer].shape) == 1:
                # Conv/FC Bias & BN Bias, Weight, Running Mean, Running Var
                assert unaligned_model[layer].shape[0] == len(alignments[layer]), f"Alignment Shape Error in {layer}: {unaligned_model[layer].shape[0]} vs {len(alignments[layer])}"
                aligned_model[layer] = torch.empty(size=unaligned_model[layer].shape,
                                                   device=unaligned_model[layer].device,
                                                   dtype=unaligned_model[layer].dtype)
                for i in range(len(alignments[layer])):
                    aligned_model[layer][i] = copy.deepcopy(unaligned_model[layer][alignments[layer][i]])
            elif len(unaligned_model[layer].shape) == 2:
                # FC Weight
                if layer == last_layer_weight:
                    # Last Layer Weight
                    assert unaligned_model[layer].shape[1] == len(alignments[layer]), f"Alignment Shape Error in {layer}: {unaligned_model[layer].shape[1]} vs {len(alignments[layer])}"
                    aligned_model[layer] = torch.empty(size=unaligned_model[layer].shape,
                                                       device=unaligned_model[layer].device,
                                                       dtype=unaligned_model[layer].dtype)
                    for i in range(len(alignments[layer])):
                        aligned_model[layer][:,i] = copy.deepcopy(unaligned_model[layer][:,alignments[layer][i]])
                else:
                    # Other FC Not Last Layer
                    assert unaligned_model[layer].shape[0] == len(alignments[layer]), f"Alignment Shape Error in {layer}: {unaligned_model[layer].shape[0]} vs {len(alignments[layer])}"
                    aligned_model[layer] = torch.empty(size=unaligned_model[layer].shape,
                                                       device=unaligned_model[layer].device,
                                                       dtype=unaligned_model[layer].dtype)
                    for i in range(len(alignments[layer])):
                        aligned_model[layer][i] = copy.deepcopy(unaligned_model[layer][alignments[layer][i]])
            elif len(unaligned_model[layer].shape) == 4:
                # Conv Weight
                assert unaligned_model[layer].shape[0] == len(alignments[layer]), f"Alignment Shape Error in {layer}: {unaligned_model[layer].shape[0]} vs {len(alignments[layer])}"
                aligned_model[layer] = torch.empty(size=unaligned_model[layer].shape,
                                                   device=unaligned_model[layer].device,
                                                   dtype=unaligned_model[layer].dtype)
                if layer == first_layer_weight:
                    for i in range(len(alignments[layer])):
                        aligned_model[layer][i] = copy.deepcopy(unaligned_model[layer][alignments[layer][i]])
                else:
                    prev = AlignedAvg._get_previous_layer(unaligned_model, layer)
                    for i in range(len(alignments[layer])):
                        for j in range(len(alignments[prev])):
                            aligned_model[layer][i, j, :, :] = copy.deepcopy(
                                unaligned_model[layer][alignments[layer][i], alignments[prev][j], :, :])


        client.update_nn_parameters(aligned_model, strict=True)

    def get_alignment(self, client, local_m, global_m):
        # Function I wrote for matched averaging
        global_model = copy.deepcopy(global_m)
        local_model = copy.deepcopy(local_m)
        alignment_map = OrderedDict()
        for i, layer in enumerate(global_model.keys()):
            # Last weight layer is second to last layer
            if i+2 == len(global_model.keys()):
                pass
            elif len(re.findall('((?:conv|fc)\d\.weight)', layer)) == 1:
                if i > 0:
                    new_layer = torch.empty(size=local_model[layer].shape,
                                            device=local_model[layer].device,
                                            dtype=local_model[layer].dtype)
                    assert len(alignment) == local_model[layer].shape[1], "Downstream Effect Mismatch"
                    for j in range(len(alignment)):
                        if len(local_model[layer].shape) == 4:
                            new_layer[:,j,:,:] = copy.deepcopy(local_model[layer][:,alignment[j],:,:])
                        elif len(local_model[layer].shape) == 2:
                            new_layer[:,j] = copy.deepcopy(local_model[layer][:,alignment[j],:,:])
                    local_model[layer] = copy.deepcopy(new_layer)
                weight_local = local_model[layer]
                shape = weight_local.shape
                weight_global = global_model[layer]
                assert shape == weight_global.shape, f"Local and Global Model Mismatch in {layer}: {shape} vs {weight_global.shape}"
                if ('use_bias' in self.__dict__.keys()) and (self.use_bias == True):
                    bias_local = local_model[layer.split('.')[0] + ".bias"]
                    if i > 0:
                        new_layer = torch.empty(size=bias_local.shape,
                                                device=bias_local.device,
                                                dtype=bias_local.dtype)
                        assert len(alignment) == new_layer.shape[0], "Downstream Effect Mismatch in Bias Layer"
                        for j in range(len(alignment)):
                            new_layer[j] = bias_local[alignment[j]]
                        bias_local = copy.deepcopy(new_layer)
                    agg_local = torch.cat((weight_local.view(shape[0], -1), bias_local.view(shape[0], -1)), 1)
                    bias_global = global_model[layer.split('.')[0] + ".bias"]
                    agg_global = torch.cat((weight_global.view(shape[0], -1), bias_global.view(shape[0], -1)), 1)
                else:
                    agg_local = weight_local.view(shape[0], -1)
                    agg_global = weight_global.view(shape[0], -1)
                alignment = AlignedAvg._align_matrix(agg_local, agg_global, **self.__dict__)
            # We go through layers in order and only update on weight so alignment preserved for subsequent BN and bias
            alignment_map[layer] = alignment
        AlignedAvg._update_alignment(client, alignment_map)

    def run_aggregation(self, epoch, client_ids, local_clients, global_client):
        self.epoch = epoch
        self.current_round = client_ids

        uploaded = self.get_uploaded_params()

        self.args.get_logger().info("Aggregating client parameters using {}", str(self))
        new_params = self.aggregate(local_clients, uploaded, global_client.get_nn_parameters())

        self.update_params(local_clients, global_client, new_params)
