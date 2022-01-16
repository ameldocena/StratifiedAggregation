

"""
Steps:
1. Record workers in each stratum.
2. Collect gradients from strata.
3. Apply BR technique in each stratum.
4. Collect aggregated gradients
5. Apply weighted (or penalized) average as the
    aggregated gradient/parameter.
"""

"""
PO Implementation:
1. We borrow functions from krum.py
2. We create a standalone python script for this one

PO Edit get_uploaded_gradients:
> We input as parameter the workers selected per stratum;
These strata should be stored separatelly within one container

> Then within those stratum we collect their gradients
> We then apply BR technique
> We then do a weighted average
    Knowledge/insight from this averaging may be derived from
    knowing which strata have less effectiveness/significance
    and where the poisoned workers are more likely to occur.
    
    Rather: We give more weight to strata that are significant
    and less likely to contain poisoned workers.
"""

"""
PO: Implementation

1. Inherit from MultiKrum class
2. Override the get_uploaded_gradients method
    > This should be based on workers in each stratum
    > This will have an input 
3. Override aggregate
    > Aggregate within each stratum
    > We then do a penalized or weighted average
4. update_params, update_global_params?, run_aggregation
    > On run_aggregation, we may need to replace self.current_round
        the stratified client_ids, (i.e., we store each client id for 
        each stratum separately).
"""
from .krum import MultiKrum
from .trimmed_mean import TrimmedMean
from .median import Median
from restricted_sharing import get_uploaded_params
from .aggregator_base import Aggregator
from .fedavg import FedAvg
import numpy as np
from clients import Global_Client

class StratifiedAggr(MultiKrum):
    #We just inherit from MultiKrum since all of its helper functions apply to
    #TrimmedMean, Median, and FedAvg
    def __init__(self, arguments, name, kwargs):
        #kwargs['NUM_WORKERS_PER_ROUND'] = kwargs['assumed_workers_per_round_stratum'] #We use proportion for each of the stratum
        #kwargs['ASSUMED_POISONED_WORKERS_PER_ROUND'] = kwargs["assumed_poisoned_per_round_stratum"]
        super().__init__(arguments, name, kwargs)
        print("Aggregator name:", self.name)
        if self.name == "StratTrimMean":
            self.beta = kwargs['TRIM_PROPORTION']
        print("Download: {}. Upload: {}".format(self.portion_downloaded, self.portion_uploaded))

    def set_num_layers(self, gradient_list):
        self.num_layers = len(gradient_list)

    def get_uploaded_gradients(self):
        #Considers restricted upload of gradients
        worker_grad_dict = dict()
        for stratum in self.strata:
            worker_grad_dict[stratum] = dict()
            for worker in self.current_round[stratum]:
                worker_grad_dict[stratum][worker] = MultiKrum.get_gradient(self, worker, self.epoch, self.portion_uploaded)
        # we retrieve the number of layers using the last iteration; could be any iteration actually
        self.set_num_layers(worker_grad_dict[stratum][worker])
        #print("Num layers:", self.num_layers)

        return worker_grad_dict

    def collect_uploaded_params(self):
        "Gets uploaded parameters of workers in each stratum; considers restricted upload sharing"
        worker_params_dict = dict()
        for stratum in self.strata:
            worker_params_dict[stratum] = list()
            for worker in self.current_round[stratum]:
                worker_params_dict[stratum].append(get_uploaded_params(self.args, worker, self.epoch, self.save)) #Does self.args here come from arguments? Yes, since we call super()

        return worker_params_dict

    """
    Two steps:
    1. Collect BR gradients per stratum
    2. Take their weighted average
    These can be turned into two functions actually
    """

    def stratified_aggregate_grads(self, uploaded_grads, strata_weights_list, global_client):
        """
        uploaded_grads - dict: output of get_uploaded_gradients/collect_uploaded_params
        """

        #Step 1. Collect the gradients/parameters per stratum
        aggregated_iters = dict() #aggregated iterates: params or grads
        for stratum in self.strata:
            #We can probably turn this into parametric where the aggregator is a parameter
            if self.name == "StratKrum":
                print("StratKrum is selected")
                aggregated_iters[stratum] = MultiKrum.aggregate(self, uploaded_grads[stratum])
            elif self.name == "StratTrimMean":
                print("StratTrimMean is selected")
                aggregated_iters[stratum] = TrimmedMean.aggregate(self, uploaded_grads[stratum])
            elif self.name == "StratMedian":
                print("StratMedian is selected")
                aggregated_iters[stratum] = Median.aggregate(self, uploaded_grads[stratum])
            elif self.name == "StratFedAvg":
                print("StratFedAvg is selected")
                uploaded_parameters = uploaded_grads[stratum]
                aggregated_parameters = Aggregator.aggregate(self, uploaded_parameters, global_client.get_nn_parameters())
                aggregated_iters[stratum] = list(aggregated_parameters.values())
                #print("Stratum: {}. Aggr params: {}.".format(stratum, aggregated_grads[stratum].keys()) )

        #Step 2. Weighted (or penalized) average of gradients/parameters for all the strata
        #This will be changed into a method
        tuned_strata_iterates = self.dynamic_tuning(aggregated_iters, strata_weights_list, global_client)

        return tuned_strata_iterates

    def strat_weight_ave(self, aggregated_iterates, strata_weights):
        """
        Computes the weighted average of strata iterates (params or grads)

        :param strata_weights: strata weights
        :param aggregated_grads: strata gradients
        :return:
        """
        stratified_aggr = list()
        for layer in range(self.num_layers):
            components_grad = list()
            # print("Strata:", self.strata, strata_weights)
            # print("uploaded_grads:", aggregated_grads.keys(), type(aggregated_grads['strata1']))

            for stratum in self.strata:
                component = strata_weights[stratum] * aggregated_iterates[stratum][layer]
                components_grad.append(component)
            weighted_grad = sum(components_grad)  # weighted average of gradients/params from all strata given the layer
            stratified_aggr.append(weighted_grad)  # append the weighted gradient/param for the given layer
        return stratified_aggr

    def dynamic_tuning(self, strata_iterates, strata_weights_list, global_client):
        """
        Tunes the strata weights dynamically, choosing the weights that yield to the greatest test performance

        :return:
        """
        global_model = global_client.get_nn_parameters()

        #Step 1. Matrix multiply strata_grads and strata_weights
        weighted_strata_iterates = list()
        test_set_perf = list()
        for strata_weights in strata_weights_list:
            weight_ave = self.strat_weight_ave(strata_iterates, strata_weights)
            weighted_strata_iterates.append(weight_ave)

            dummy_client = Global_Client(global_client.args, global_client.test_data_loader)

            if self.name == "StratFedAvg": #parameter-based aggregation technique
                weight_ave_params = dict(zip(list(global_model.keys()), weight_ave))
                dummy_client.update_nn_parameters(weight_ave_params)
            else: #We are using the update on gradient-based aggregation techniques
                self.update_global_params(dummy_client, weight_ave, save = False)

        #Step 2. Compute the test set performance for each iter in Step 1
        #How? We create a dummy client to test the performance of each candidate params
        #We then measure the test set performance for a given weighted average params

            results = dummy_client.test(dummy = True)[0] #Global test accuracy as our performance metric
            test_set_perf.append(results)

        #Step 3. Return the best stratum weight, and do the aggreation
        print("Strata weights list:", strata_weights_list)
        print("Test set perfs:", test_set_perf)
        max_perf = np.argmax(np.asarray(test_set_perf))
        tuned_weights = strata_weights_list[max_perf] #tuned weights

        print("Max weights:", max_perf, tuned_weights)
        tuned_iterates = self.strat_weight_ave(strata_iterates, tuned_weights) #tuned weighted iterates

        return tuned_iterates

    def run_aggregation(self, epoch, stratified_workers, strata_weights, local_clients, global_client):
        """
        stratified_workers - dict: stratum as keys while worker ids within each
                                stratum stored as list as the corresponding values
        """
        self.epoch = epoch
        self.current_round = stratified_workers
        self.strata = self.current_round.keys() #Different strata
        global_model = global_client.get_nn_parameters()
        self.num_layers = len(list(global_model.keys()))
        #print("NUm_layers:", self.num_layers, global_model.keys())

        #We can insert the dynamic tuning here

        if self.name=='StratFedAvg':
            uploaded_params = self.collect_uploaded_params()
            self.args.get_logger().info("Aggregating STRATIFIED client parameters using {}", str(self))

            # Stratified aggregation parameters
            stratified_aggr_params = self.stratified_aggregate_grads(uploaded_params, strata_weights, global_client)
            # Update params
            stratified_aggr_params = dict(zip(list(global_model.keys()), stratified_aggr_params))
            # Change to FedAvg.update_params to account for restricted sharing; if no restricted sharing, this just works fine
            Aggregator.update_params(self, local_clients, global_client, stratified_aggr_params)

        else:
            uploaded_gradients = self.get_uploaded_gradients()
            self.args.get_logger().info("Aggregating STRATIFIED client gradients using {}", str(self))

            #Stratified aggregation gradients
            stratified_aggr_gradient = self.stratified_aggregate_grads(uploaded_gradients, strata_weights, global_client)
            #Model parameters
            self.update_params(local_clients, global_client, stratified_aggr_gradient)