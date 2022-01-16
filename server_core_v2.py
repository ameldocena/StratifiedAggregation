from loguru import logger
from saving_utils import generate_experiment_ids, convert_results_to_csv, save_results
from arguments import Arguments
from dataloaders import load_data_loader, generate_data_loaders_from_distributed_dataset
from clients import create_clients, Global_Client
import data_distribution_v2 as data_distribution
from utilities import select_aggregator, count_poisoned_stratum
from stratified_random_sampling_v2 import StratifiedRandomSampling
import random as rd
import numpy as np
import pickle

def train_subset_of_clients(data_disbn, root_disbn, epoch, args, clients, exp_id, poisoned_workers, global_model, aggregator, strata_weights=None, worker_disbn = None):
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

    #THIS PART HERE NEEDS TO BE SELECTED WHETHER IT COULD NOT BE RANDOMIZED FOR KRUM
    #Q: How do we randomize clients along with poisoned workers?
    #1: Given k poisoned workers, we ensure at each round we sample m = k + benign?
    #2: Or we say benign + k, then sample m workers?
    #3: Or we sample alpha*m from benign, then (1 - alpha)*m from poisoned?

    #Something that needs to be answered
    #Probably: We get the round worker selection strategy, which is an object of its own
    #We then select workers for that round by the select_round_workers method;
    #Turns out to be simple random sampling.

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

        #This is if we aggregate the entire strata
        if (aggregator.name != "StratKrum") and (aggregator.name != "StratTrimMean") and (aggregator.name != "StratMedian") and (aggregator.name != "StratFedAvg"):
            print("Stratified selection list:", aggregator.name)

            #Where is the part where we stratify the workers. Is this embedded in select_round_workers?
            random_workers = selection.select_round_workers(root_disbn, poisoned_workers, kwargs['influence'], worker_disbn)[0] #list of stratified random workers
            print("Random workers:", random_workers)
            args.get_logger().info("Commencing experiment #{}, epoch #{}", str(exp_id), str(epoch))

            for client_idx in random_workers:
                args.get_logger().info("Training epoch #{} on client #{}", str(epoch),
                                       str(clients[client_idx].get_client_index()))
                clients[client_idx].train(epoch)
                clients_results[client_idx] = clients[client_idx].test()

                #This is where we should insert the testing of the performance

        #This one is if we aggregate within each stratum
        else:
            print("Stratified selection dict:", aggregator.name)
            random_workers = selection.select_round_workers(root_disbn, poisoned_workers, kwargs['influence'], worker_disbn)[1] #dict of stratified random workers per stratum
            count_poisoned_stratum(random_workers, poisoned_workers)
            print("Random workers:", random_workers)
            args.get_logger().info("Commencing experiment #{}, epoch #{}", str(exp_id), str(epoch))

            for stratum in random_workers:
                for client_idx in random_workers[stratum]:
                    args.get_logger().info("Training epoch #{} on client #{}", str(epoch),
                                           str(clients[client_idx].get_client_index()))
                    clients[client_idx].train(epoch)
                    clients_results[client_idx] = clients[client_idx].test()
    # ###Here, I can store the results/performance of local clients per epoch
    # client_results = dict()
    # for client_idx in random_workers:
    #     client_results[client_idx] = clients[client_idx].test()

    #Here is where the aggregation is done for this epoch
    #I need to make sure my devised run_aggregation for Krum is compatible
    #Q: Why do we need the clients and random_workers together?
    #Ah okay, ranom_workers is probably the index to select among clients

    #We can do an if-statement here: If non-stratified, random_workers would be
    #a list
    #If stratified, then random_workers would be a dictionary of stratum workers
    #Okay: Turns out no need for an if statement then?
    #It will still be random_workers, but in its generation we do the conditional statement
    #If aggregator is stratified, then we return stratified_workers
    #Else, return a list of random_workers
    if (aggregator.name != "StratKrum") and (aggregator.name != "StratTrimMean") and (aggregator.name != "StratMedian") and (aggregator.name != "StratFedAvg"):
        print("Regular aggregation is selected")
        aggregator.run_aggregation(epoch, random_workers, clients, global_model)
    else:
        print("Stratified aggregation")
        aggregator.run_aggregation(epoch, random_workers, strata_weights, clients, global_model)
    #What the run_aggregation returns is that it updates the global parameters and the downloadable parameters
    #Here what is done is that it measures the test accurcacy of the global model and then returns it along with num_workers

    #IMPROVEMENT: WE MAY NEED THE RANDOM_WORKERS AS LIST
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
    args.set_randomized_start(initialized == "Randomized")
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

    #Apply attribute skew on test_data_loader to simulate non-IID
    #What does a dataloader comprise of?


    #Here data_distribution is a module. Where was it defined?
    workers_disbn = data_distribution.identify_random_elements(args.get_num_workers(), KWARGS['ndistr'], KWARGS['pois_distr'])
    print("Workers disbn:", workers_disbn)
    poisoned_workers = workers_disbn['poisoned']
    skewed_workers = workers_disbn['skewed']

    print("Skewed workers:", skewed_workers)
    print("Poisoned workers:", poisoned_workers)
    # Distribute batches equal volume IID
    if distribution == "IID":
        distributed_train_dataset = data_distribution.distribute_batches_equally(train_data_loader, args.get_num_workers())
    elif distribution == "Non-IID":
        print("Non-IId is chosen")
        distributed_train_dataset = data_distribution.distribute_non_iid(args.get_batch_size(), train_data_loader, args.get_num_workers(), poisoned_workers, poisoned_class)
        print("Len Non-IID:", len(distributed_train_dataset))
    elif distribution == "Non-IID_v2":
        distributed_train_dataset = data_distribution.distribute_non_iid_v2(args.get_batch_size(), train_data_loader, args.get_num_workers(), poisoned_workers, poisoned_class, skewed_workers)

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
                                                                                n_classes=10, p=0.5, scalar=1.5)
        else:
            distributed_train_dataset = data_distribution.distribute_label_skew(args.get_batch_size(),
                                                                                train_data_loader,
                                                                                args.get_num_workers(),
                                                                                poisoned_workers, poisoned_class, q=0.1,
                                                                                n_classes=10, p=0.5, scalar=1.5)
    elif distribution == "Label and Quantity Skew":
        if poisoned_class == -1:
            distributed_train_dataset = data_distribution.distribute_label_skew(args.get_batch_size(),
                                                                                train_data_loader,
                                                                                args.get_num_workers(), [], 0, q=0.5,
                                                                                n_classes=10, p=0.5, scalar=1.5)
        else:
            distributed_train_dataset = data_distribution.distribute_label_skew(args.get_batch_size(),
                                                                                train_data_loader,
                                                                                args.get_num_workers(),
                                                                                poisoned_workers, poisoned_class, q=0.5,
                                                                                n_classes=10, p=0.5, scalar=1.5)

    #Data an root distribution used for stratification
    #if KWARGS['SELECTION_STRATEGY'] == "Stratified_RS":
    data_disbn = data_distribution.compile_data(distributed_train_dataset, poisoned_workers, poisoned_class, target, KWARGS['conceal_pois_class']) #This data_disbn hasn't switched the poisoned class to the target class yet
    root_disbn = data_distribution.generate_root_disbn(args.get_batch_size(), train_data_loader)
    #root_disbn = data_distribution.generate_root_midpt_disbn((0, 1), shape = (28, 28), nlabels = 10)
        #We can have a condition here where poisoned workers report their true scores
    #else:
    # data_disbn = None
    # root_disbn = None

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
        #So this just trains a subset of clients.
        #We subset an entire pool of clients of both benign and poisoned workers.

        #The results here is the test set results and workers_selected
        #What does train_subset_of_clients do?
        #This part here is where random selection of workers occurs.
        #EDIT: WE CAN ADD THE DISTRIBUTION DATA SET HERE
        #PO1: We create a function that stratify the workers. And then we just call stratified RS which samples each sample
        #PO2: Embedded function for stratified RS
        strata_weights = KWARGS['strata_weights']

        #This part here we do sample selection of workers and then do local training
        results, workers_selected, clients_results = train_subset_of_clients(data_disbn, root_disbn, epoch, args, clients, idx, poisoned_workers, global_model, aggregator, strata_weights, worker_disbn = workers_disbn)
        print("Workers_selected:", workers_selected)
        epoch_test_set_results.append(results)
        if KWARGS['SELECTION_STRATEGY'] == "Stratified_RS":
            collect_workers = list()
            for stratum in workers_selected.keys():
                collect_workers.extend(workers_selected[stratum])
            worker_selection.append(collect_workers)
        else: #Worker selection is done by SRS
            worker_selection.append(workers_selected)
        print(worker_selection)

        #Collect workers selected for test performance at given epoch
        collect_client_results['epoch_' + str(epoch)] = clients_results

    results = convert_results_to_csv(epoch_test_set_results)
    save_results(results, results_files[0])
    save_results(worker_selection, worker_selections_files[0])

    #Pickle dump collected local client model performance
    with open(str(idx) + '_local_clients_results.pkl', 'wb') as f:
        pickle.dump(collect_client_results, f)

    logger.remove(handler)
