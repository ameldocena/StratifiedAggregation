from loguru import logger
from saving_utils import generate_experiment_ids, convert_results_to_csv, save_results
from arguments import Arguments
from dataloaders import load_data_loader, generate_data_loaders_from_distributed_dataset
from clients import create_clients, Global_Client
import data_distribution
from utilities import select_aggregator, count_poisoned_stratum, generate_uniform_weights
from stratified_random_sampling_v3 import StratifiedRandomSampling
import random as rd
import numpy as np
import pickle

def train_subset_of_clients(data_disbn, root_disbn, epoch, args, clients, exp_id, poisoned_workers, global_model, aggregator, filter_thresh, strata_weights=None, worker_disbn = None):
    """
    Train a subset of clients per round

    :param epoch: epoch
    :type epoch: int
    :param args: arguments
    :type args: Arguments
    :param clients: clients
    :type clients: list(Client)
    :param poisoned_workers: indices of poisoned workers
    :type poisoned_workers: list(int)
    :param global_model: the global model
    :param aggregator: the aggregator
    """
    kwargs = args.get_round_worker_selection_strategy_kwargs()
    kwargs["current_epoch_number"] = epoch

    #Saves starting global_model at given epoch
    global_model.save_model(epoch=epoch, suffix='start')

    #This is where the Stratified random sampling comes into place
    clients_results = dict() #Collection of worker model test performance
    if kwargs['SELECTION_STRATEGY']=="Simple_RS":
        random_workers = args.get_round_worker_selection_strategy().select_round_workers(
            list(range(args.get_num_workers())),
            poisoned_workers,
            kwargs) #Note that the select_round_workers does not use poisoned_workers. It merely does a simple random sampling
        #on the workers index [0, 1, ..., args.get_num_workers-1]
        print("Random workers:", random_workers)
        args.get_logger().info("Commencing experiment #{}, epoch #{}", str(exp_id), str(epoch))

        for client_idx in random_workers:
            args.get_logger().info("Training epoch #{} on client #{}", str(epoch),
                                   str(clients[client_idx].get_client_index()))
            clients[client_idx].train(epoch)
            clients_results[client_idx] = clients[client_idx].test()

    elif kwargs['SELECTION_STRATEGY']=="Stratified_RS":
        selection = StratifiedRandomSampling(data_disbn, kwargs)
        #This part here: The stratified_workers should also be returned
        #That is, the workers per stratum

        #This is if we aggregate the entire strata. For instance, stratified random sampling but the aggregation is FedAvg
        if (aggregator.name != "StratKrum") and (aggregator.name != "StratTrimMean") and (aggregator.name != "StratMedian") and (aggregator.name != "StratFedAvg"):
            print("Stratified selection list:", aggregator.name)

            #Where is the part where we stratify the workers. Is this embedded in select_round_workers?
            workers_list = list(range(args.get_num_workers()))
            random_workers = selection.select_round_workers(root_disbn, poisoned_workers, workers_list, kwargs['influence'], worker_disbn)[0] #list of stratified random workers
            print("Random workers:", random_workers)
            args.get_logger().info("Commencing experiment #{}, epoch #{}", str(exp_id), str(epoch))

            for client_idx in random_workers:
                args.get_logger().info("Training epoch #{} on client #{}", str(epoch),
                                       str(clients[client_idx].get_client_index()))
                clients[client_idx].train(epoch)
                clients_results[client_idx] = clients[client_idx].test()

            # This is where we should insert the testing of the performance to compute the fscore
            # We then stratify the workers based on the fscore

            #This part here we insert the Stage 2 unsupervised stratification
            #I know my value!
            #We compute the F-scores of each client. We can import teh function from Jupyter notebook

            ##F-score
            fscores = selection.compute_fscores(clients_results)
            skewed_fscore = selection.fscore_skewedlabel(fscores)
            #average_fscore = selection.average_fscore(fscores)

            filter_local_updates = selection.stratify_stage2(skewed_fscore, random_workers, threshold=filter_thresh)

            ##Precision and recall
            # skewed_precision = selection.precision_skewedlabel(clients_results)
            # skewed_recall = selection.recall_skewedlabel(clients_results)
            # prec_rec_scores = {'precision' : skewed_precision, 'recall': skewed_recall}
            #
            # filter_local_updates = selection.stratify_stage2(prec_rec_scores, random_workers, threshold = filter_thresh)

            print("Stage 1:", random_workers)
            print("Stage 2:", filter_local_updates)
            #Stage 3 of stratification: Intersection of Stage 1 and Stage 2
            #Set as random_workers, selecting [1] as a dict of selected workers
            random_workers = selection.select_round_workers_stage2(random_workers, filter_local_updates)[0]

            print("Stage 3:", random_workers)

        #This one is if we aggregate within each stratum and do stratified aggregation. For instance, stratified RS and then stratified FedAvg
        else:
            print("Stratified selection dict:", aggregator.name)

            #Stage 1 - Unsupervised stratification on non-IID attributes
            workers_list = list(range(args.get_num_workers()))
            s1_results = selection.select_round_workers(root_disbn, poisoned_workers, workers_list, kwargs['influence'], worker_disbn)
            random_workers_list_s1 = s1_results[0] #list of stratified random workers per stratum
            random_workers_dict_s1 = s1_results[1]  #dict of stratified random workers per stratum
            count_poisoned_stratum(random_workers_dict_s1, poisoned_workers)
            print("Random workers:", random_workers_list_s1)
            args.get_logger().info("Commencing experiment #{}, epoch #{}", str(exp_id), str(epoch))

            for stratum in random_workers_dict_s1:
                for client_idx in random_workers_dict_s1[stratum]:
                    args.get_logger().info("Training epoch #{} on client #{}", str(epoch),
                                           str(clients[client_idx].get_client_index()))
                    clients[client_idx].train(epoch)
                    clients_results[client_idx] = clients[client_idx].test()

            #This part here we insert the Stage 2 unsupervised stratification
            #I know my value!
            #We compute the F-scores of each client. We can import teh function from Jupyter notebook
            #PO2: We use the skewed label precision and recall as the stratification; and then the F-score of the
            #precision and recall centroids as the filter
            fscores = selection.compute_fscores(clients_results)
            skewed_fscore = selection.fscore_skewedlabel(fscores)
            # #average_fscore = selection.average_fscore(fscores)
            filter_local_updates = selection.stratify_stage2(skewed_fscore, random_workers_list_s1, threshold = filter_thresh)

            ##Precision and recall
            # skewed_precision = selection.precision_skewedlabel(clients_results)
            # skewed_recall = selection.recall_skewedlabel(clients_results)
            # prec_rec_scores = {'precision': skewed_precision, 'recall': skewed_recall}
            #
            # filter_local_updates = selection.stratify_stage2_v2(prec_rec_scores, random_workers_list_s1, threshold=filter_thresh)
            print("Stage 1:", random_workers_dict_s1)
            print("Stage 2:", filter_local_updates)
            #Stage 3 of stratification: Intersection of Stage 1 and Stage 2
            #Set as random_workers, selecting [1] as a dict of selected workers
            random_workers = selection.select_round_workers_stage2(random_workers_dict_s1, filter_local_updates)[1]
            print("Stage 3:", random_workers)

            count_poisoned_stratum(random_workers, poisoned_workers)
            """
            It would be good to know how many in each stratum are poisoned updates after the two-stage stratification
            """

    if (aggregator.name != "StratKrum") and (aggregator.name != "StratTrimMean") and (aggregator.name != "StratMedian") and (aggregator.name != "StratFedAvg"):
        print("Regular aggregation is selected")
        aggregator.run_aggregation(epoch, random_workers, clients, global_model)
    else:
        print("Stratified aggregation")

        #We pop out the empty stratum
        nonempty_random_workers = dict()

        #We store non-empty random workers and strata weights
        for stratum in random_workers:
            if len(random_workers[stratum]) > 0:
                nonempty_random_workers[stratum] = random_workers[stratum]

        #Generate Uniform weights
        strata_weights = [generate_uniform_weights(nonempty_random_workers)]


        # for strata in strata_weights: #for strata weights stored in strata_weights_list
        #     strata_weights_dummy = dict()
        #     for stratum in random_workers:
        #         if len(random_workers[stratum]) > 0:
        #             strata_weights_dummy[stratum] = strata[stratum]
        #     nonempty_strata_weights.append(strata_weights_dummy)

        # for stratum in list(random_workers):
        #     if len(random_workers[stratum]) == 0:
        #         random_workers.pop(stratum) #We pop that stratum from the workers dictionary
        #         for strata in strata_weights_list: #We pop the stratum from all the strata weights stored in the list
        #             strata.pop(stratum)
        print("Random workers post:", nonempty_random_workers)
        print("Strata weights post:", strata_weights)
        aggregator.run_aggregation(epoch, nonempty_random_workers, strata_weights, clients, global_model)
        random_workers = nonempty_random_workers.copy()
    #What the run_aggregation returns is that it updates the global parameters and the downloadable parameters
    #Here what is done is that it measures the test accurcacy of the global model and then returns it along with num_workers

    return global_model.test(), random_workers, clients_results

def run_exp(replacement_method, num_poisoned_workers, KWARGS, client_selection_strategy, idx, uploaded, downloaded, dataset, initialized, distribution, poisoned_class, target, agg_name):
    """
    Modified version of run_exp from the original work
    :param replacement_method: function to poison data
    :param num_poisoned_workers: int number of poisoned workers
    :param KWARGS:
    :param client_selection_strategy:
    :param idx:
    :param uploaded:
    :param downloaded:
    :param dataset:
    :param initialized:
    :param distribution:
    :param poisoned_class:
    :return:
    """
    log_files, results_files, models_folders, worker_selections_files = generate_experiment_ids(idx, 1)

    # Initialize logger
    handler = logger.add(log_files[0], enqueue=True)

    args = Arguments(logger)
    args.switch_CIFAR_MNIST(dataset)
    args.set_randomized_start(initialized == "Randomized") #What is this?
    args.set_model_save_path(models_folders[0])
    args.set_num_poisoned_workers(num_poisoned_workers)

    #Here we set the round worker selection strategy
    args.set_round_worker_selection_strategy_kwargs(KWARGS)
    args.set_client_selection_strategy(client_selection_strategy)
    args.set_portion_uploaded(uploaded)
    args.set_portion_downloaded(downloaded)
    args.log()

    train_data_loader = load_data_loader(logger, args, test=False)
    test_data_loader = load_data_loader(logger, args, test=True)

    #Randomly assign poisoned workers
    poisoned_workers = data_distribution.identify_random_elements(args.get_num_workers(),
                                                                  args.get_num_poisoned_workers())
    print("Poisoned workers:", poisoned_workers)

    # Distribute batches equal volume IID
    if distribution == "IID":
        distributed_train_dataset = data_distribution.distribute_batches_equally(train_data_loader, args.get_num_workers())

    elif distribution == "Non-IID":
        print("Non-IId is chosen")
        distributed_train_dataset = data_distribution.distribute_non_iid(args.get_batch_size(), train_data_loader, args.get_num_workers(), poisoned_workers, poisoned_class)

    elif distribution == "Non-IID_v2":
        distributed_train_dataset = data_distribution.distribute_non_iid_v2(args.get_batch_size(), train_data_loader, args.get_num_workers(), poisoned_workers, poisoned_class, skewed_workers = None)

    elif distribution == "Label Skew":
        if poisoned_class == -1:
            distributed_train_dataset = data_distribution.distribute_label_skew(args.get_batch_size(),
                                                                                train_data_loader,
                                                                                args.get_num_workers(), [], 0, q=0.5,
                                                                                n_classes=10, p=0, scalar=1)
        else:
            distributed_train_dataset = data_distribution.distribute_label_skew(args.get_batch_size(),
                                                                                train_data_loader,
                                                                                args.get_num_workers(),
                                                                                poisoned_workers, poisoned_class, q=0.5,
                                                                                n_classes=10, p=0, scalar=1)
    elif distribution == "Quantity Skew":
        if poisoned_class == -1:
            distributed_train_dataset = data_distribution.distribute_label_skew(args.get_batch_size(),
                                                                                train_data_loader,
                                                                                args.get_num_workers(), [], 0, q=0.1,
                                                                                n_classes=10, p=0.5, scalar=2.0)
        else:
            distributed_train_dataset = data_distribution.distribute_label_skew(args.get_batch_size(),
                                                                                train_data_loader,
                                                                                args.get_num_workers(),
                                                                                poisoned_workers, poisoned_class, q=0.1,
                                                                                n_classes=10, p=0.5, scalar=2.0)
    elif distribution == "Label and Quantity Skew":
        if poisoned_class == -1:
            distributed_train_dataset = data_distribution.distribute_label_skew(args.get_batch_size(),
                                                                                train_data_loader,
                                                                                args.get_num_workers(), [], 0, q=0.5,
                                                                                n_classes=10, p=0.5, scalar=2.0)
        else:
            distributed_train_dataset = data_distribution.distribute_label_skew(args.get_batch_size(),
                                                                                train_data_loader,
                                                                                args.get_num_workers(),
                                                                                poisoned_workers, poisoned_class, q=0.5,
                                                                                n_classes=10, p=0.5, scalar=2.0)

    #Data an root distribution used for stratification
    print("Distributed data:", type(distributed_train_dataset))
    print("Length distributed data:", len(distributed_train_dataset))
    for i in distributed_train_dataset: print(len(i))

    data_disbn = data_distribution.compile_data(distributed_train_dataset, poisoned_workers, poisoned_class, target, KWARGS['conceal_pois_class']) #This data_disbn hasn't switched the poisoned class to the target class yet
    root_disbn = data_distribution.generate_root_disbn(args.get_batch_size(), train_data_loader)
    #root_disbn = data_distribution.generate_root_midpt_disbn((0, 1), shape = (28, 28), nlabels = 10)
        #We can have a condition here where poisoned workers report their true scores

    data_distribution.store_summary(idx, data_disbn, poisoned_workers, root_disbn)
    print("Stored data summary")
    distributed_train_dataset = data_distribution.convert_distributed_data_into_numpy(distributed_train_dataset)

    #Poison data
    distributed_train_dataset = data_distribution.poison_data(logger, distributed_train_dataset, args.get_num_workers(), poisoned_workers, replacement_method, poisoned_class, target)
    """
    Steps above:
    1. Identify who the poisoned workers
    2. Determine how the distribution of data will be
    3. The poison_data merely had their data labeled as 1 flipped to label = 9.
    """

    #Generate Torch DataLoaders for the distributed data
    train_data_loaders = generate_data_loaders_from_distributed_dataset(distributed_train_dataset, args.get_batch_size())

    #Creates clients class for the entire number of workers; note poisoned wokers included
    clients = create_clients(args, train_data_loaders, test_data_loader)
    #Create global client class
    global_model = Global_Client(args, test_data_loader) #Retrieve the present global model

    #Select the aggergator
    aggregator = select_aggregator(args, agg_name, KWARGS)

    args.set_aggregator(aggregator)

    #IDEA IMPLEMENTATION: This part here we insert the stratified Byzantine-robust aggregation technique

    #Recording the results
    epoch_test_set_results = []
    worker_selection = []

    collect_client_results = dict() #Collect client results per epoch

    for epoch in range(1, args.get_num_epochs() + 1):
        #Set seed: If we are to re-run into multiple experiments, we need a variable that changes here per experimental run
        seed = epoch*10 + epoch
        rd.seed(seed)
        np.random.seed(seed)
        strata_weights = KWARGS['strata_weights']

        #This part here we do sample selection of workers and then do local training
        if KWARGS["stratify_kwargs_s2"] is None:
            thresh = None
        else:
            thresh = KWARGS["stratify_kwargs_s2"]['thresh']
        results, workers_selected, clients_results = train_subset_of_clients(data_disbn, root_disbn, epoch, args, clients, idx, poisoned_workers, global_model, aggregator, thresh, strata_weights)
        print("Workers_selected:", workers_selected)
        epoch_test_set_results.append(results)
        if KWARGS['SELECTION_STRATEGY'] == "Stratified_RS":
            collect_workers = list()
            for stratum in workers_selected.keys():
                collect_workers.extend(workers_selected[stratum])
            worker_selection.append(collect_workers)
        else: #Worker selection is done by SRS
            worker_selection.append(workers_selected)

        #Collect workers selected for test performance at given epoch
        collect_client_results['epoch_' + str(epoch)] = clients_results

    results = convert_results_to_csv(epoch_test_set_results)
    save_results(results, results_files[0])
    save_results(worker_selection, worker_selections_files[0])

    #Pickle dump collected local client model performance
    with open(str(idx) + '_local_clients_results.pkl', 'wb') as f:
        pickle.dump(collect_client_results, f)

    logger.remove(handler)
