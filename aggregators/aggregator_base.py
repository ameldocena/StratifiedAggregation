from collections import OrderedDict
from restricted_sharing import get_downloadable_params, get_uploaded_params

class Aggregator:
    def __init__(self, arguments, name, save=False, **kwargs):
        self.args = arguments
        self.name = name
        self._epoch = 1
        self._current_round = []
        self.save = save
        self.__dict__.update(kwargs)

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, epoch):
        self._epoch = epoch

    @property
    def current_round(self):
        return self._current_round

    @current_round.setter
    def current_round(self, current_round):
        expected_num = self.args.get_round_worker_selection_strategy_kwargs()["NUM_WORKERS_PER_ROUND"]
        #assert len(current_round) == expected_num, f"Expected {expected_num} workers, got {len(current_round)}"
        self._current_round = current_round

    def __str__(self):
        return self.name # fix in run_exp

    def get_uploaded_params(self):
        return [get_uploaded_params(self.args, idx, self.epoch, self.save) for idx in self.current_round]

    def get_downloadable_params(self):
        return get_downloadable_params(self.args, self.epoch, self.save)

    #Shall I insert here the uploaded_gradients method from restricted parameters sharing?
    #I can also insert or import this from restricted parameter sharing, and then insert/call from krum
    #After all, I will overwrite run_aggregation in there.

    def aggregate(self, local_models, global_model):
        new_params = OrderedDict()
        for layer in global_model.keys():
            #This part here: Why are we using .data?
            #for i in local_models:
            #    print(type(i[layer].data), i[layer].data)
            new_params[layer] = sum([i[layer].data for i in local_models])/len(local_models)
        return new_params

    def update_params(self, local_clients, global_client, new_params):
        self.args.get_logger().info("Updating global model parameters")
        global_client.update_nn_parameters(new_params)
        global_client.save_model(epoch=self.epoch, suffix='end')
        downloadable = self.get_downloadable_params()
        for client in local_clients:
            self.args.get_logger().info("Updating parameters on client #{}", str(client.get_client_index()))
            client.update_nn_parameters(downloadable)

    #Well we shall update this part
    def run_aggregation(self, epoch, client_ids, local_clients, global_client):
        self.epoch = epoch
        self.current_round = client_ids

        uploaded = self.get_uploaded_params()

        self.args.get_logger().info("Aggregating client parameters using {}", str(self))
        new_params = self.aggregate(uploaded, global_client.get_nn_parameters())

        self.update_params(local_clients, global_client, new_params)
