import numpy as np
from .aggregator_base import Aggregator
from .krum import MultiKrum
from .fedavg import FedAvg
from scipy import stats

class TrimmedMean(Aggregator):
    """
    Implementation of the Trimmed Mean byzantine-robust aggregation technique.
    """
    def __init__(self, arguments, name, kwargs):
        print("Trimmed Mean is SELECTED")
        super().__init__(arguments, name)
        self.beta = kwargs['TRIM_PROPORTION'] #We set here the proportion to cut the tails and heads of the distribution
        self.portion_uploaded = self.args.get_portion_uploaded()
        self.portion_downloaded = self.args.get_portion_downloaded()

        # Some assertions for restricted portions
        assert self.portion_uploaded > 0 and self.portion_uploaded <= 1, "Upload proportion should be within (0, 1]"
        assert self.portion_downloaded > 0 and self.portion_downloaded <= 1, "Download proportion should be within (0, 1]"

    def restricted_grad_sharing(self, gradients, portion):
        #Restricts upload of gradient by the stipulated portion
        #Function borrowed from MultiKrum.
        return MultiKrum.restricted_grad_sharing(self, gradients, portion)

    def get_gradient(self, idx, epoch, portion):
        #Function borrowed from MultiKrum.
        # idx - worker index
        # epoch - communication round
        # This function collects the uploaded gradients of clients for a given epoch, and converts these from Torch to numpy
        return MultiKrum.get_gradient(self, idx, epoch, portion)

    def get_uploaded_gradients(self):
        """
        Function borrowed from MultiKrum
        Collects the gradients of each worker given the epoch.

        Returns: Worker and its gradients (list)
        """
        return MultiKrum.get_uploaded_gradients(self)

    def aggregate(self, uploaded_gradients):
        """
        Computes the Trimmed-Mean gradient"
        Step 1. Collect the gradients of selected workers for each layer.
        Step 2. Calculate the trimmed_mean per layer, given the beta parameter (the proportion to trim)
        """
        #Q: WHAT IF WE SET THIS AS NONE?
        assert len(uploaded_gradients.values()) > 0, "Empty list of gradient to aggregate"
        num_layers = len(uploaded_gradients[list(uploaded_gradients.keys())[0]])  # number of layers

        trim_mean_gradient = list()
        for layer in range(num_layers):
            collect_grads = list()
            for i in uploaded_gradients.keys():
                collect_grads.append(uploaded_gradients[i][layer])
            trim_mean_layer = stats.trim_mean(np.array(collect_grads), self.beta)
            trim_mean_gradient.append(trim_mean_layer)
        print("Return: Trimmed mean gradient")
        return trim_mean_gradient

    def update_global_params(self, global_client, agg_gradient):
        "Updates global parameters"
        return MultiKrum.update_global_params(self, global_client, agg_gradient)

    def update_params(self, local_clients, global_client, agg_gradient):
        #A. Update global parameters
        updated_global_params = MultiKrum.update_global_params(self, global_client, agg_gradient)

        #B. Update local parameters
        print("Portion downloaded by clients:", self.portion_downloaded)
        for client in local_clients:
            ##If download = 1.0, update all local clients with full updated_params
            ##Else, use FedAvg.simple_update_with_missing to account for restricted download of paramaters
            if self.portion_downloaded == 1:
                self.args.get_logger().info("Unrestricted download of global parameters on client #{}",
                                            str(client.get_client_index()))
                client.update_nn_parameters(updated_global_params)

            else:
                downloadable = super().get_downloadable_params()  # Restricted global parameters for download
                FedAvg.simple_update_with_missing(self, client, downloadable)

    def run_aggregation(self, epoch, client_ids, local_clients, global_client):
        MultiKrum.run_aggregation(self, epoch, client_ids, local_clients, global_client)