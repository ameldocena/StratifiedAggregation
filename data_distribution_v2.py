import random
import numpy as np
import math
import albumentations as A
import torch
from torch.utils.data import TensorDataset, DataLoader
import copy
import pickle

def replace_N_with_M(targets, source=1, new=9):
    #What does this do? It replaces all labels = 1 to target labels = 9?
    #If so, then this works for data set that have up to 10 labels.
    #print("Poisoned class: {}. Target: {}.".format(source, new))
    for idx in range(len(targets)):
        if targets[idx] == source:
            targets[idx] = new

    return targets

#Function that selects the original and skewed workers (distribution or data shards)
#And then selects poisoned workers from each shard
def identify_random_elements(max, num_random_elements):
    """
    Picks a specified number of random elements from 0 - max.
    Used to select poisoned workers
    Unmodified from Tolpegin et al.

    :param max: Max number to pick from
    :type max: int
    :param num_random_elements: Number of random elements to select
    :type num_random_elements: int
    :return: list
    """
    if num_random_elements > max:
        return []

    ids = []
    x = 0
    while x < num_random_elements:
        rand_int = random.randint(0, max - 1)

        if rand_int not in ids:
            ids.append(rand_int)
            x += 1

    return ids

def identify_random_elements_attr(num_workers, data_distrib, poisoned_distrib):
    """
    Selects workers for actual data distribution. Here we only have two actual distribution: the original and skewed.
    But we can extend this to include more number of distribution

    :param num_workers (int): Number (or pool) of workers to choose from for data distribution
    :param data_distrib (int): Number of data shards (or distributed)
    :param poisoned_distrib (dict): data_distrib as keys; number (of poisoned workers) as values
    :return:
    """
    dict_distrib = dict()

    pool = list(range(num_workers)) #pool to take from
    to_take = math.ceil(len(pool)/data_distrib) #number of workers to take
    original = random.sample(pool, to_take) #workers with original data
    skewed = list(set(range(num_workers)) - set(original)) #workers with skewed data

    print("Orig:", original)
    print("Skewed:", skewed)
    print("To-take:", to_take)
    poisoned = list()
    pois_orig = random.sample(original, poisoned_distrib['orig']) #inject poisoned workers among the originals
    pois_skewed = random.sample(skewed, poisoned_distrib['skewed']) #inject poisoned among the skewed
    poisoned.extend(pois_orig)
    poisoned.extend(pois_skewed)

    dict_distrib['orig'] = original
    dict_distrib['skewed'] = skewed
    dict_distrib['poisoned'] = poisoned

    return dict_distrib

def distr_equal_propn(label_data, num_workers):
    """
    Idea:
    1. For each label, distribute disproportionate allocation to workers.
    2. Apply the worker's allocation to label_data and store in distr_labeldata,
        where the keys are workers and the values are the labeled data with X and Y.

    Inputs:
    label_data - dict: output of segregate_labels
    num_workers - scalar: number of workers
    """

    #Step 1: Distribute allocation to workers
    distr_propn = dict() #A dict of dicts: labels and then worker allocations
    labels = label_data.keys()

    #Initial allocation
    for label in labels:
        ndata = len(label_data[label]['X']) #number of data points for the given label
        propn = [ndata // num_workers] * num_workers
        distr_propn[label] = dict(zip(list(range(num_workers)), propn))
        assert round(sum(propn), 1) <= ndata, "Allocation of proportions should at most be the length of label data"

    return distr_propn

def distr_equal_data(label_data, num_workers):
    #Step: Apply the workers allocation to label_data and store in distr_labeldata

    labels = label_data.keys()
    distr_labeldata = dict()

    distr_propn = distr_equal_propn(label_data, num_workers) #distribute equal proportion to workers

    for_distr = copy.deepcopy(label_data)
    for worker in range(num_workers):
        distr_labeldata[worker] = dict()
        total_data = 0
        for label in labels:
            distr_labeldata[worker][label] = dict()
            slice_data = distr_propn[label][worker]
            #print("worker: {}. slice_data: {}".format(worker, slice_data))

            distr_labeldata[worker][label]['X'] = for_distr[label]['X'][:slice_data]
            distr_labeldata[worker][label]['Y'] = for_distr[label]['Y'][:slice_data]

            #Adjust the available data for
            for_distr[label]['X'] = for_distr[label]['X'][slice_data: ]
            for_distr[label]['Y'] = for_distr[label]['Y'][slice_data: ]
            total_data += len(distr_labeldata[worker][label]['X'])
        print("Worker: {}. Allocated data: {}".format(worker, total_data))
    return distr_labeldata

def distribute_batches_equally(train_data_loader, num_workers):
    """
    Gives each worker the same number of batches of training data.
    Unmodified from original work

    :param train_data_loader: Training data loader
    :type train_data_loader: torch.utils.data.DataLoader
    :param num_workers: number of workers
    :type num_workers: int
    """
    distributed_dataset = [[] for i in range(num_workers)]

    for batch_idx, (data, target) in enumerate(train_data_loader):
        worker_idx = batch_idx % num_workers

        distributed_dataset[worker_idx].append((data, target))

    return distributed_dataset

def distribute_non_iid(batch_size, train_data_loader, num_workers, poisoned_ids, poisoned_class, p=0.5, n_classes=10):
    '''
    Distributes training data with bias
    Requires an equal number of instances in each class
    :param train_data_loader:
    :param num_workers:
    :param poisoned_ids:
    :return:
    '''
    #("Batch size:", batch_size)
    # First we need to assign all of the workers to groups, with as many groups as there are classes
    distributed_dataset = [[] for i in range(num_workers)]
    L_groups = {}
    used_workers = []
    workers_per_class = int(num_workers / n_classes)
    # If no workers are poisoned, we just arbitrarily call some of them poisoned here, their data will not actually be poisoned
    if len(poisoned_ids) == 0:
        poisoned_ids = random.sample(list(range(num_workers)), workers_per_class)
    L_groups[poisoned_class] = []
    if len(poisoned_ids) / num_workers == 1 / n_classes:
        # If 10% of workers are poisoned (assuming 10 classes), we put all of the poisoned workers in
        # the group corresponding to the poisoned class
        L_groups[poisoned_class].extend(poisoned_ids)
        used_workers.extend(poisoned_ids)
    elif len(poisoned_ids) / num_workers < 1 / n_classes:
        # If there are fewer poisoner workers than there are workers that should be assigned to the poisoned class
        # we assign all of the poisoned workers to the poisoned class and randomly select benign workers to make up the difference
        L_groups[poisoned_class].extend(poisoned_ids)
        used_workers.extend(poisoned_ids)
        possible_workers = [i for i in range(num_workers) if i not in used_workers]
        benign = random.sample(possible_workers, k=workers_per_class - len(poisoned_ids))
        L_groups[poisoned_class].extend(benign)
        used_workers.extend(benign)
    else:
        # If there are more poisoned workers than there are workers per class, we randomly select some of the
        # poisoned workers to be in the poisoned class group and the rest will randomly be assigned to other groups
        poisoned_biased = random.sample(poisoned_ids, workers_per_class)
        L_groups[poisoned_class] = poisoned_biased
        used_workers.extend(poisoned_biased)

    # For the rest of the groups, we randomly select workers that have not already been assigned to the poisoned class group
    for l in range(n_classes):
        if l == poisoned_class:
            continue
        L_groups[l] = []
        selected_workers = random.sample([i for i in range(num_workers) if i not in used_workers], workers_per_class)
        L_groups[l].extend(selected_workers)
        used_workers.extend(selected_workers)

    train_data_classes = {class_num: [] for class_num in range(n_classes)}
    # Next we need to separate out the data samples and group them by target class
    samples_per_class = {class_num: 0 for class_num in range(n_classes)}
    for batch_idx, (data, target) in enumerate(train_data_loader):
        for sample in range(len(target)):
            ix = int(target[sample].numpy())
            train_data_classes[ix].append((data[sample], target[sample]))
            samples_per_class[ix] += 1
    """
    train_data_classes contains the samples for each the classes/labels.
    """

    # And then we go through each data grouping and assign portion p of them to the group corresponding to that class
    # the rest of the samples for each class grouping is randomly assigned to the other classes, as evenly as possible
    grouped_data = {class_num: [] for class_num in range(n_classes)}
    for group in train_data_classes.keys():
        random.shuffle(train_data_classes[group])
        topx = int(round(p * len(train_data_classes[group])))
        grouped_data[group].extend(train_data_classes[group][:topx])
        leftover = len(train_data_classes[group][topx:]) // (n_classes - 1)

        # This should ensure that the remainders will be distributed across classes such that the total number of samples in each group is the same
        remainder_val = sum([1 for _ in range(len(train_data_classes[group][topx:]) % (n_classes - 1))])
        remainder = [0 for _ in range(group + 1)] + [1 for _ in range(remainder_val)] + [0 for _ in range(
            n_classes - group - remainder_val)]
        post = remainder[n_classes:]
        pre = remainder[len(post):n_classes]
        remainder = post + pre

        last_stop = topx
        for i in range(n_classes):
            if i == group:
                continue
            grouped_data[i].extend(train_data_classes[group][last_stop:last_stop + leftover + remainder[i]])
            last_stop += leftover + remainder[i]

    # Now that the data has been assigned to each group, we randomly distribute it across the workers in the group

    for worker in range(num_workers):
        for group in range(n_classes):
            if worker in L_groups[group]:
                selected = random.sample(list(range(len(grouped_data[group]))),
                                         k=int(samples_per_class[group] / workers_per_class))
                temp = [grouped_data[group][i] for i in selected]
                # This would need to be changed if the number of samples is not divisible by batch size
                worker_vals = []
                for i in range(len(temp) // batch_size):
                    ix = i * batch_size
                    vals = temp[ix:ix + batch_size]
                    targets = []
                    inputs = []
                    for j in vals:
                        targets.append(int(j[1].numpy()))
                        inputs.append(j[0].numpy())
                    worker_vals.append((torch.Tensor(inputs), torch.Tensor(targets)))
                distributed_dataset[worker].extend(worker_vals)
                grouped_data[group] = [grouped_data[group][i] for i in range(len(grouped_data[group])) if
                                       i not in selected]

    return distributed_dataset

def convert_distributed_data_into_numpy(distributed_dataset):
    """
    Converts a distributed dataset (returned by a data distribution method) from Tensors into numpy arrays.
    Unchanged from original work
    :param distributed_dataset: Distributed dataset
    :type distributed_dataset: list(tuple)
    """
    converted_distributed_dataset = []

    for worker_idx in range(len(distributed_dataset)):
        worker_training_data = distributed_dataset[worker_idx]

        X_ = np.array([tensor.cpu().numpy() for batch in worker_training_data for tensor in batch[0]])
        Y_ = np.array([tensor.cpu().numpy() for batch in worker_training_data for tensor in batch[1]])

        converted_distributed_dataset.append((X_, Y_))

    return converted_distributed_dataset

def poison_data(logger, distributed_dataset, num_workers, poisoned_worker_ids, replacement_method, poisoned_class, target):
    """
    Poison worker data
    Unchanged from original work
    :param logger: logger
    :type logger: loguru.logger
    :param distributed_dataset: Distributed dataset
    :type distributed_dataset: list(tuple)
    :param num_workers: Number of workers overall
    :type num_workers: int
    :param poisoned_worker_ids: IDs poisoned workers
    :type poisoned_worker_ids: list(int)
    :param replacement_method: Replacement methods to use to replace
    :type replacement_method: list(method)
    """
    poisoned_dataset = []

    """
    Edited here. We used the label_class_dict instead. Each worker has its own class labels.
    """
    label_class_dict = {}
    #print("Poisoning data")

    #We can turn class_labels into a dictionary where for each worker, the value is the list of labels

    ##print("distributed_dataset[0][1]:", distributed_dataset[0][1], list(set(distributed_dataset[0][1])))
    ##print("poison_data class_labels:", class_labels)

    logger.info("Poisoning data for workers: {}".format(str(poisoned_worker_ids)))

    for worker_idx in range(num_workers):
        if worker_idx in poisoned_worker_ids:
            # previously poisoned_dataset.append(apply_class_label_replacement(distributed_dataset[worker_idx][0], distributed_dataset[worker_idx][1], replacement_method))
            #How do you poison a data set? You just switch them?
            #What is inside distributed_dataset? It seems it is a dict that can be indexed by worker ID.
            #And it has two elements, [0] and [1]? What are these?. Probably, distributed_dataset is a tuple. Prob X and Y?
            #What is this replacement method?
            poisoned_dataset.append((distributed_dataset[worker_idx][0], replacement_method(distributed_dataset[worker_idx][1], poisoned_class, target)))
        else:
            poisoned_dataset.append(distributed_dataset[worker_idx])
        label_class_dict[worker_idx] = list(set(distributed_dataset[worker_idx][1]))

    #for worker_idx in range(num_workers):
        #print("Worker: {}. Label class set: {}".format(worker_idx, label_class_dict[worker_idx]))

    log_client_data_statistics(logger, label_class_dict, poisoned_dataset)

    return poisoned_dataset

def log_client_data_statistics(logger, label_class_dict, distributed_dataset):
    # Unchanged from original work
    """
    Logs all client data statistics.

    :param logger: logger
    :type logger: loguru.logger
    :param label_class_set: set of class labels
    :type label_class_set: list
    :param distributed_dataset: distributed dataset
    :type distributed_dataset: list(tuple)
    """
    """
    I think the problem here is that the program assumes that all workers have same number of classes.
    What it did is that it just took one worker's class labels and assumed it for all.
    For our case, each worker has own class labels.
    
    Edited: We use label_class_dict
    
    Bug: The label_class_dict[client_idx] does not match with distributed_dataset[client_idx][1], (i.e., poisoned_dataset)
    
    """
    #label_class_set
    #print("len(distributed_dataset):", len(distributed_dataset))
    ##print("label_class_set:", label_class_set)
    for client_idx in range(len(distributed_dataset)):
        #print("client_idx:", client_idx)
        label_class_set = label_class_dict[client_idx]
        client_class_nums = {class_val : 0 for class_val in label_class_set}
        #print("label_class_set:", label_class_set)
        #print("distributed_dataset[client_idx][1]:", np.unique(distributed_dataset[client_idx][1]))
        #print("client_class_nums:", client_class_nums)
        for target in distributed_dataset[client_idx][1]:
            #print("target:", type(target), target)
            client_class_nums[target] += 1

        logger.info("Client #{} has data distribution: {}".format(client_idx, str(list(client_class_nums.values()))))

"""
Data distribution part 2
This takes into account:
 i. quantity skewness,
 ii. label skewness, (not 
 iii. attribute skewness 
"""

def segregate_x_y(dataloader, batch_size):
    """
    Segregates X and Y data from a data loader.
    Note that the dataloader is divided into different shards of size batch_size.
    """
    X_data = list()
    Y_data = list()
    for _, (x, y) in enumerate(dataloader):
        #print("batch_size:", batch_size)
        for elem in range(batch_size):
            #print("(x, y) shape:", (x.shape, y.shape))
            #print("y:", y.numpy()[elem])
            #print(x[elem][0].shape)
            x_point = x.numpy()[elem]
            y_point = y.numpy()[elem]
            X_data.append(x_point)
            Y_data.append(int(y_point))

    segregated_data = dict()
    segregated_data['X'] = np.array(X_data)
    segregated_data['Y'] = np.array(Y_data)

    return segregated_data

def segregate_labels(segregated_data):
    """
    Segregates and shuffles data for each of the labels
    """
    labels = np.unique(segregated_data['Y'])
    label_data = dict()
    for label in labels:
        label_indices = np.argwhere(segregated_data['Y'] == label)  # indices in the data for a given label
        label_data[label] = dict()

        # Segregate X, Y data
        label_data[label]['X'] = list()
        label_data[label]['Y'] = list()
        #print("label indices:", label_indices.shape, label_indices)
        for index in label_indices:
            index = np.squeeze(index)
            #print("index:", index)
            label_data[label]['X'].append(segregated_data['X'][index])  # We store up X, Y data per label
            label_data[label]['Y'].append(segregated_data['Y'][index])
        label_data[label]['X'] = np.array(label_data[label]['X'])
        label_data[label]['Y'] = np.array(label_data[label]['Y'])

        #WE SHOULD SHUFFLE LABELED DATA HERE

    return label_data

#Simulate Quantity Skewness
#Volume of worker data for a given label relative to the total volume of data shared
#across workers
def distr_labeldata_unequal(label_data, num_workers):
    """
    Idea:
    1. For each label, distribute disproportionate allocation to workers.
    2. Apply the worker's allocation to label_data and store in distr_labeldata,
        where the keys are workers and the values are the labeled data with X and Y.

    Inputs:
    label_data - dict: output of segregate_labels
    num_workers - scalar: number of workers
    """

    #Step 1: Distribute allocation to workers
    distr_propn = dict() #A dict of dicts: labels and then worker allocations
    labels = label_data.keys()

    #Initial allocation
    for label in labels:
        ndata = len(label_data[label]['X']) #number of data points for the given label
        #print("Label: {}. No. data points: {}".format(label, ndata))
        remaining = 100 #100 percent
        workers = list(range(num_workers))
        w = random.choice(workers) #Pick the first worker to be allocated first
        workers.remove(w)
        propn = list()  # For sanity check. Distributed propotion should sum up 1.

        distr_propn[label] = dict()
        s = int(50 + (50/len(workers)))
        p = random.randint(1, s)
        distr_propn[label][w] = int(p/100 * ndata) #proportion of labeled data
        propn.append(p/100)

        #Allocation to intermediate workers
        remaining -= p
        while len(workers) > 1:
            w = random.choice(workers)
            workers.remove(w)
            p = random.randint(1, int(remaining/len(workers)))
            distr_propn[label][w] = int(p/100 * ndata)
            propn.append(p/100)
            remaining -= p

        #Last allocation
        w = workers.pop() #last worker to be allocated
        distr_propn[label][w] = int(remaining/100 * ndata)
        propn.append(remaining/100)

        #"Propn: {}. Sum: {}".format(propn, sum(propn)))
        #print("distribution:", distr_propn[label])
        assert round(sum(propn), 1) == 1.0, "Allocation of proportions should equal 1"
    #return distr_propn
    #Step 2: Apply the workers allocation to label_data and store in distr_labeldata
    distr_labeldata = dict()

    for_distr = copy.deepcopy(label_data)
    for worker in range(num_workers):
        distr_labeldata[worker] = dict()
        total_data = 0
        for label in labels:
            distr_labeldata[worker][label] = dict()
            slice_data = distr_propn[label][worker]
            #print("worker: {}. slice_data: {}".format(worker, slice_data))

            distr_labeldata[worker][label]['X'] = for_distr[label]['X'][:slice_data]
            distr_labeldata[worker][label]['Y'] = for_distr[label]['Y'][:slice_data]

            #Adjust the available data for
            for_distr[label]['X'] = for_distr[label]['X'][slice_data: ]
            for_distr[label]['Y'] = for_distr[label]['Y'][slice_data: ]
            total_data += len(distr_labeldata[worker][label]['X'])
        #print("Worker: {}. Allocated data: {}".format(worker, total_data))
    return distr_labeldata

# Simulate label skewness in terms of label count
# NOTE: NEEDS TO BE SEEDED FOR REPRODUCIBILITY

def skew_label_count(num_workers, nlabels):
    """
    Skews the label count for each worker; that is each worker will have at least [2, nlabels] count of labels picked randomly.
    For instance, if we randomly pick the label count q = 3, then that worker has 3 labels out of the nlabels.

    Inputs:
    num_workers - scalar: number of workers
    nlabels - scalar: number of labels

    Returns:
    label_counts - dictionary: label counts for each workewr
    """
    label_counts = dict()  # container for the label counts of each worker

    #50 percent of the workers will have all the labels, while the other half will have incomplete labels
    for worker in range(math.floor(num_workers/2)):
        label_counts[worker] = nlabels
    for worker in range(math.floor(num_workers/2), num_workers):
        label_counts[worker] = random.choice(list(range(2, nlabels)))
    return label_counts

# NOTE: NEEDS TO BE SEEDED FOR REPRODUCIBILITY.
def identify_label_skew(nlabels, skewed_label_count, num_workers, poisoned_workers, poisoned_class):
    """
    Inputs:
    skewed_label_count - dict: output of skew_label_count
    num_workers - scalar (int): number of workers
    poisoned_workers - list: list of poisoned workers
    poisoned_class - scalar (int): poisoned class (label flipping attack)

    Returns:
    identify_labels - dict: skewed labels per worker
    """
    identify_labels = dict()

    # Identify the labels for each worker.
    for worker in range(num_workers):
        ##Note that if a poisoned worker, the poisoned class should be present
        identify_labels[worker] = list()
        choose_labels = list(range(nlabels))

        ##Pick labels
        if worker in poisoned_workers:
            picked_label = poisoned_class
            identify_labels[worker].append(picked_label)  # We include the poisoned class among the labels
            choose_labels.remove(picked_label)
            for pick_labels in range(skewed_label_count[worker] - 1):
                picked_label = random.choice(choose_labels)
                identify_labels[worker].append(picked_label)
                choose_labels.remove(picked_label)

        else:
            for pick_labels in range(skewed_label_count[worker]):
                picked_label = random.choice(choose_labels)
                identify_labels[worker].append(picked_label)
                choose_labels.remove(picked_label)

        # Sanity check: Number of labels picked should match the corresponding skewed_label_count
        assert skewed_label_count[worker] == len(
            identify_labels[worker]), "Number of labels picked mismatch with supposed label count"

    for worker in range(num_workers):
        if worker in poisoned_workers:
            print("Poisoned worker: {}. Labels: {}".format(worker, identify_labels[worker]))
        else:
            print("Benign worker: {}. Labels: {}".format(worker, identify_labels[worker]))
    return identify_labels


# Simulate label skewness in terms of proportion of data.
# NOTE: NEEDS TO BE SEEDED FOR REPRODUCIBILITY
"""
Let q be the total count of labels in a workers data set, and let p be the proportion of a dominant label.
Note that we assume/require that each worker has at least two labels in its data, i.e., q>=2.
The remaining 1-p proportion is distributed randomly to the other labels. 
If the worker is regular/benign, the dominant label is taken randomly from its label pool. But if poisoned, then
the dominant label is the poisoned class. This setup is for a label-flipping attack.

We randomly pick p to be any value within [0.50, 0.50 + 0.50/(1+(q-1))].
For instance, if q = 2, then p is picked within [0.50, 0.75].
We then distribute the 1-p proportion to the other labels.
"""


def skew_label_proportion(skewed_label_count, identified_labels, num_workers, poisoned_workers, poisoned_class):
    """
    Set proportion (of total data) for each of the label.
    For a dominant label, set the proportion to be within the range [0.50, 0.50 + 0.50/(1+(m-1))].
    If a poisoned worker, the dominant label is the poisoned class.
    If regular, pick randomly from its pool of labels.

    Inputs:
    skewed_label_count - dict: output of skew_label_count()
    identified_labels - dict: output of indentify_label_skew()
    num_workers - scalar (int): number of workers
    poisoned_workers - scalar (int): no. of poisoned workers
    """
    skewed_label_propn = dict()
    for worker in range(num_workers):
        #print("\n Worker:", worker)
        propn = list()  # container for the proportions and should sum up to 1 (will be used for sanity check)
        m = skewed_label_count[worker]  # number of labels for a worker
        assert m>0, "Number of labels for a worker should be greater than 0."
        p = (random.randint(50, math.floor(50 + (50 / m)))) / 100  # proportion of dominant label
        #print("Proportion:", p)
        propn.append(p)
        skewed_label_propn[worker] = dict()  # container for proportion for each label in worker's data
        choose_labels = identified_labels[worker].copy()
        #print("worker: {}. choose_labels: {}. propn: {}".format(worker, choose_labels, propn))

        # Set the proportion for the dominant label.
        ##If a poisoned worker, we set the poisoned class as the dominant label
        if worker in poisoned_workers:
            dom_label = poisoned_class
        else:
            dom_label = random.choice(choose_labels)
        skewed_label_propn[worker][dom_label] = p  # Store dominant label and proportion
        choose_labels.remove(dom_label)  # We remove the poisoned class from the labels to choose from
        #print("dominant label:", dom_label)
        #print("choose_labels: {}. propn: {}".format(choose_labels, propn))

        # Distribute proportion to remaining labels.
        s = 100 * (1 - p)  # remaining proportion to choose from
        #print("s:", s)

        #print("number of iterations:", len(choose_labels))
        # for iteration in range(len(choose_labels)):

        while len(choose_labels) > 1:
            # print("iteration:", iteration)
            picked_label = random.choice(choose_labels)
            remaining_labels = len(choose_labels)
            #print("remaining_labels:", remaining_labels)
            b = random.randint(1, math.floor(s / remaining_labels))  # We pick a proportion within (0.01, remaining propn/remaining labels)
            #print("b:", b)
            skewed_label_propn[worker][picked_label] = round(b / 100, 2)

            propn.append(round(b / 100, 2))

            # Adjustments
            #excess = math.floor(s / remaining_labels) - b
            s = s - b  # remaining proportion
            choose_labels.remove(picked_label)  # remaining labels to choose from
            #print("adj s:", s)
            #print("choose_labels: {}. propn: {}. sum: {}. diff: {}".format(choose_labels, propn, sum(propn),
            #                                                               1 - sum(propn)))

        # We allocate the last remaining propn to the last remaining label
        picked_label = choose_labels.pop()
        skewed_label_propn[worker][picked_label] = round(s / 100, 2)
        propn.append(round(s / 100, 2))
        print("\nPropn: {}. Sum: {}.".format(propn, sum(propn)))

        # Sanity check: The total proportion of labels for each worker should sum up to 1
        #print("Propn: {}. Sum: {}.".format(propn, sum(propn)))
        assert round(sum(propn), 1) == 1.0, "Proportion should equal 1"

    return skewed_label_propn


"""
UPNEXT: Do the allocation per worker
1. For each of the data per label of each worker, we apply the proportion

Inputs: 
> skewed_label_prop - dict: {keys: worker, values: (label, propn)}
> distr_labeldata - dict: {keys: worker, values: dict{keys: label, values: (X, Y)}}
"""


def apply_skewed_propn(distr_labeldata, skewed_label_prop):
    workers = distr_labeldata.keys()
    #print(workers)
    applied_skewed = dict()  # worker data where skewed proportion has been applied
    for worker in workers:
        labels_present = skewed_label_prop[worker].keys()  # identified skewed labels
        #print("worker: {}. labels_present: {}.".format(worker, labels_present))
        applied_skewed[worker] = dict()
        total_data = 0
        for label in labels_present:
            slice_data = int(skewed_label_prop[worker][label] * len(distr_labeldata[worker][label]['X']))
            #print("Label: {}. Sliced data propn: {}".format(label, slice_data))
            applied_skewed[worker][label] = dict()
            applied_skewed[worker][label]['X'] = distr_labeldata[worker][label]['X'][:slice_data]
            applied_skewed[worker][label]['Y'] = distr_labeldata[worker][label]['Y'][:slice_data]
            total_data += len(applied_skewed[worker][label]['X'])
            #print("Length of data: {}. Length of sliced data: {}".format(len(distr_labeldata[worker][label]['X']), len(applied_skewed[worker][label]['X'])))
        #print("Total data:", total_data)

    return applied_skewed


"Note that the CIFAR10 data have been normalized. We may have to denormalize it first."
"""
NOTES:
1. Needs to be seeded for reproducibility.
2. Needs albumentations to be imported as A

"""

#aug_list=[A.VerticalFlip(p=1.0), A.GaussNoise(always_apply=True, p=1.0)]

def apply_attribute_skew(distributed_data, skewed_workers):
    """
    Applies horizontal flip or color jitter to a sampled workers' data.

    Inputs:
    distrtibuted_data - dict: distributed data for all the workers
    nsample - sample of workers whose data would be jittered/rotated
    aug_list - list of albumentations (package) data augmentation functions with set parameters
    """

    aug_images = copy.deepcopy(distributed_data)  # container for augmented images
    for worker in skewed_workers:
        mlabels = distributed_data[worker].keys()  # labels in workers data
        for label in mlabels:
            #images = aug_images[worker][label]['X'].copy()
            #transform = A.Compose(aug_list)  # We apply data augmentation
            #aug_images[worker][label]['X'] = transform(image=images)['image']

            images = list()
            for datapt in aug_images[worker][label]['X']:
                images.append(datapt.T.reshape(datapt.shape))
            aug_images[worker][label]['X'] = np.asarray(images)
        #print("L2 Norm:", np.linalg.norm(distributed_data[worker][label]['X']))
        #print("L2 Norm:", np.linalg.norm(aug_images[worker][label]['X']))
        print("Attr skew applied:", np.linalg.norm(aug_images[worker][label]['X'] - distributed_data[worker][label]['X']))
    return aug_images

def shuffle_data_shards(aug_images, batch_size):
    """
    applied_skewed_propn - dict: output of apply_attribute_skew
    batch_size - scalar: batch size
    """

    workers = list(aug_images.keys())
    #print("Workers:", workers)
    shuffled_data_shards = [[] for i in range(len(workers))] #Container as list of lists

    for w in range(len(workers)):
        worker = workers[w] #
        mlabels = aug_images[worker].keys() #number of labels in workers data
        worker_data = dict()
        worker_data['X'] = list() #container for worker's X data
        worker_data['Y'] = list() #container for worker's Y data

        #This part we collect the X and Y data of worker that are stored separately
        #by labels in aug_images
        for label in mlabels:
            worker_data['X'].extend(aug_images[worker][label]['X'])
            worker_data['Y'].extend(aug_images[worker][label]['Y'])

        #Shuffle worker X and Y data here
        shuffled_worker_data = copy.deepcopy(worker_data)

        #Divide shuffled data into shards of batch_size
        length_data = len(worker_data['X'])
        num_shards = math.floor(length_data/batch_size) + int(length_data % batch_size >= 1) #number of shards
        alloc_shards = 0 #number of allocated shards
        start = 0
        stop = batch_size
        while (num_shards - alloc_shards) > 1:
            shard_X = np.array(shuffled_worker_data['X'][start: stop])
            shard_Y = np.array(shuffled_worker_data['Y'][start: stop])
            shuffled_data_shards[w].append((torch.from_numpy(shard_X), torch.from_numpy(shard_Y)))

            #Adjust iterables/parameters
            start = stop
            stop = start + batch_size
            alloc_shards += 1

        #Allocate all remaining data to the last remaining shard
        shard_X = np.array(shuffled_worker_data['X'][start:])
        shard_Y = np.array(shuffled_worker_data['Y'][start:])
        shuffled_data_shards[w].append((torch.from_numpy(shard_X), torch.from_numpy(shard_Y)))

    return shuffled_data_shards

def distribute_non_iid_v2(batch_size, train_data_loader, num_workers, poisoned_ids, poisoned_class, skewed_workers, n_classes=10):
    #Note we added skewed_workers here
    train_data = segregate_x_y(train_data_loader, batch_size)
    label_data = segregate_labels(train_data)
    distr_labeldata = distr_equal_data(label_data, num_workers) #distribute equally
    aug_images = apply_attribute_skew(distr_labeldata, skewed_workers)
    shuffled_data_shards = shuffle_data_shards(aug_images, batch_size)
    return shuffled_data_shards

"""
Brody's Codes: Label skew and Quantity skew
"""

def distribute_label_skew(batch_size, train_data_loader, num_workers, poisoned_ids, poisoned_class, q=0.5, n_classes=10, p=0, scalar=1.5):
    '''
    Distributes training data with label bias, then passes to distribute_quantity_skew
    Requires an equal number of instances in each class
    :param batch_size: the batch size used for training
    :param train_data_loader: the training dataloader
    :param num_workers: the total number of. workers
    :param poisoned_ids: a list of the poisoned workers if applicable
    :param poisoned_class: an integer corresponding to the source class for the label flipping attack
    :param q: the label skew parameter, following notation in FLTrust. The portion of samples for any given label that
        will be controlled by the group of clients skewed toward that label. If n_classes = 10 and q=0.1 data are IID
    :param n_classes: the number of classes, default 10
    :param p: the portion of clients that will receive quantity skew as well. If clients should not have quantity skew,
        use p=0
    :param scalar: the amount of quantity skew to apply to selected clients. If scalar=1.5, the workers that are chosen
        to have high quantity will have 1.5x as many data instances as the workers with low quantity skew
    :return: the distributed dataset
    '''
    # First we need to assign all of the workers to groups, with as many groups as there are classes
    distributed_dataset = [[] for i in range(num_workers)]
    L_groups = {}
    used_workers = []
    workers_per_class = int(num_workers / n_classes)
    # If no workers are poisoned, we just arbitrarily call some of them poisoned here, their data will not be poisoned
    if len(poisoned_ids) == 0:
        poisoned_ids = random.sample(list(range(num_workers)), workers_per_class)
    L_groups[poisoned_class] = []
    if len(poisoned_ids) / num_workers == 1 / n_classes:
        # If 10% of workers are poisoned (assuming 10 classes), we put half of the poisoned workers in
        # the group corresponding to the poisoned class
        temp = random.sample(poisoned_ids, k=int(workers_per_class/2))
        L_groups[poisoned_class].extend(temp)
        used_workers.extend(temp)
        possible_workers = [i for i in range(num_workers) if i not in used_workers]
        benign = random.sample(possible_workers, k=workers_per_class - len(temp))
        L_groups[poisoned_class].extend(benign)
        used_workers.extend(benign)
    elif len(poisoned_ids) / num_workers < 1 / n_classes:
        # If there are fewer poisoner workers than there are workers that should be assigned to the poisoned class
        # we assign poisoned workers to the poisoned class to fill half of that group and randomly select benign workers
        # to make up the difference. The remaining poisoned workers will be randomly assigned to other groups
        L_groups[poisoned_class].extend(poisoned_ids)
        used_workers.extend(poisoned_ids)
        possible_workers = [i for i in range(num_workers) if i not in used_workers]
        benign = random.sample(possible_workers, k=workers_per_class - len(poisoned_ids))
        L_groups[poisoned_class].extend(benign)
        used_workers.extend(benign)
    else:
        # If there are more poisoned workers than there are workers per class, we randomly select some of the
        # poisoned workers to be in the poisoned class group and the rest will randomly be assigned to other groups
        poisoned_biased = random.sample(poisoned_ids, k=int(workers_per_class/2))
        L_groups[poisoned_class] = poisoned_biased
        used_workers.extend(poisoned_biased)
        possible_workers = [i for i in range(num_workers) if i not in poisoned_ids]
        benign = random.sample(possible_workers, k=workers_per_class - len(poisoned_biased))
        L_groups[poisoned_class].extend(benign)
        used_workers.extend(benign)

    # For the rest of the groups, we randomly select workers that have not already been assigned
    for l in range(n_classes):
        if l == poisoned_class:
            continue
        L_groups[l] = []
        selected_workers = random.sample([i for i in range(num_workers) if i not in used_workers], workers_per_class)
        L_groups[l].extend(selected_workers)
        used_workers.extend(selected_workers)

    train_data_classes = {class_num: [] for class_num in range(n_classes)}
    # Next we need to separate out the data samples and group them by target class
    samples_per_class = {class_num: 0 for class_num in range(n_classes)}
    for batch_idx, (data, target) in enumerate(train_data_loader):
        for sample in range(len(target)):
            ix = int(target[sample].numpy())
            train_data_classes[ix].append((data[sample], target[sample]))
            samples_per_class[ix] += 1
    # And then we go through each data grouping and assign portion p of them to the group corresponding to that class
    # the rest of the samples for each class grouping is randomly assigned to the other classes, as evenly as possible
    grouped_data = {class_num: [] for class_num in range(n_classes)}
    for group in train_data_classes.keys():
        random.shuffle(train_data_classes[group])
        topx = int(round(q * len(train_data_classes[group])))
        grouped_data[group].extend(train_data_classes[group][:topx])
        leftover = len(train_data_classes[group][topx:]) // (n_classes - 1)

        # This should ensure that the remainders will be distributed across classes such that the total number of
        # samples in each group is the same
        remainder_val = sum([1 for _ in range(len(train_data_classes[group][topx:]) % (n_classes - 1))])
        remainder = [0 for _ in range(group + 1)] + [1 for _ in range(remainder_val)] + [0 for _ in range(
            n_classes - group - remainder_val)]
        post = remainder[n_classes:]
        pre = remainder[len(post):n_classes]
        remainder = post + pre

        last_stop = topx
        for i in range(n_classes):
            if i == group:
                continue
            grouped_data[i].extend(train_data_classes[group][last_stop:last_stop + leftover + remainder[i]])
            last_stop += leftover + remainder[i]

    # Now that the data has been assigned to each group, we randomly distribute it across the workers in the group
    return distribute_quantity_skew(batch_size, grouped_data, distributed_dataset, L_groups, p, scalar)

def distribute_quantity_skew(batch_size, grouped_data, distributed_dataset, groupings, p=0.5, scalar=1.5):
    """
    Adds quantity skew to the data distribution. If p=0. or scalar=1., no skew is applied and the data are divided
        evenly among the workers in each label group.
    :param batch_size: the batch size for training
    :param grouped_data: a dictionary containing the data for each label skew group, key is the label integer and value
        is the data
    :param distributed_dataset: an initialized empty dictionary that will be filled with data for each worker
    :param groupings: a dictionary of the groupings for each worker id, key is the label integer and value is a list of
        worker ids
    :param p: the portion of workers within each group that will receive higher data quantities, p=0 indicates no skew
    :param scalar: the factor used to multiply the size of datasets for high quantity workers, e.g. if scalar=1.5 then
        each worker with high quantity skew has 1.5x as many data points as the low quantity workers in their group
    :return: the distributed dataset
    """
    for n, group in groupings.items():
        high_quantity = random.sample(group, k=int(p*len(group)))
        low_quantity = [i for i in group if i not in high_quantity]
        base_k = int(len(grouped_data[n])/len(group))
        print(f"Base K: {base_k}")
        print(f"Length of grouped data: {len(grouped_data[n])}")
        if p > 0.:
            low_k = int(len(grouped_data[n]) / (len(low_quantity) + len(high_quantity) * scalar))
            high_k = int(low_k * scalar)
            print(f"High Quantity Skew: {high_quantity}")
            print(f"High Quantity K: {high_k}")
            print(f"Low Quantity Skew: {low_quantity}")
            print(f"Low Quantity K: {low_k}")
        else:
            low_k = base_k
            assert len(high_quantity) == 0, "Quantity skew with probability 0 should have no high quantity clients"
            print(f"High Quantity Skew: {high_quantity}")
            print(f"Low Quantity Skew: {low_quantity}")
            print(f"Base K: {base_k}")
        for worker in high_quantity:
            selected = random.sample(list(range(len(grouped_data[n]))), k=high_k)
            temp = [grouped_data[n][i] for i in selected]
            # This would need to be changed if the number of samples is not divisible by batch size
            worker_vals = []
            for i in range(len(temp) // batch_size):
                ix = i * batch_size
                vals = temp[ix:ix + batch_size]
                targets = []
                inputs = []
                for j in vals:
                    targets.append(int(j[1].numpy()))
                    inputs.append(j[0].numpy())
                worker_vals.append((torch.Tensor(inputs), torch.Tensor(targets)))
            distributed_dataset[worker].extend(worker_vals)
            grouped_data[n] = [grouped_data[n][i] for i in range(len(grouped_data[n])) if i not in selected]
        for nx, worker in enumerate(low_quantity):
            if nx+1 == len(low_quantity):
                print(f"Length of remaining data = {len(grouped_data[n])}\nLow_k = {low_k}")
                temp = grouped_data[n]
            else:
                selected = random.sample(list(range(len(grouped_data[n]))), k=low_k)
                temp = [grouped_data[n][i] for i in selected]
            # This would need to be changed if the number of samples is not divisible by batch size
            worker_vals = []
            for i in range(len(temp) // batch_size):
                ix = i * batch_size
                vals = temp[ix:ix + batch_size]
                targets = []
                inputs = []
                for j in vals:
                    targets.append(int(j[1].numpy()))
                    inputs.append(j[0].numpy())
                worker_vals.append((torch.Tensor(inputs), torch.Tensor(targets)))
            distributed_dataset[worker].extend(worker_vals)
            if nx+1 != len(low_quantity):
                grouped_data[n] = [grouped_data[n][i] for i in range(len(grouped_data[n])) if i not in selected]
    return distributed_dataset



def store_summary(idx, compiled_data, poisoned_workers, root_disbn = None):
    """
    Store summary statistics of generated distribution of workers

    :param idx: experiment index
    :param compiled_data: distribution of data per worker, per label
    :param poisoned_workers: list of poisoned workers
    :param root_disbn: Root training distribution, if provided
    :return: summary count of distribution stored as Excel file
    """
    store = dict()
    store['pois_workers'] = poisoned_workers
    store['data'] = compiled_data
    store['root_disbn'] = root_disbn
    with open('summary_' + str(idx) + '.pkl', 'wb') as f:
        pickle.dump(store, f)

def generate_root_disbn(batch_size, train_data_loader):
    train_data = segregate_x_y(train_data_loader, batch_size)
    label_data = segregate_labels(train_data)

    # Root distribution: Average training data for each of the labels
    root_disbn = dict()
    for label in label_data.keys():
        root_disbn[label] = np.squeeze(np.mean(label_data[label]['X'], axis=0))
        #print("Label: {}. Shape: {}".format(label, root_disbn[label].shape))
    return root_disbn

def generate_root_midpt_disbn(rng, shape, nlabels):
    """
    Returns a root distribution whose values are the midpoint of the pixel range
    :param range: pixel range, tuple
    :param shape: shape of the array
    :param nlabels: number of labels
    :return: (dict) root midpoint for each label
    """
    root_disbn = dict()
    mid_pt = (rng[0] + rng[1])/2

    for label in range(nlabels):
        root_disbn[label] = np.full(shape, mid_pt)

    return root_disbn


def compile_data(shuffled_data_shards, poisoned_workers, poisoned_class, target_class, conceal_pois_class = True):
    """
    This function compiles X and Y data for each worker from a starting data that is
    per worker, per label, then X and Y.

    Inputs:
    shuffled_data_shards - list: output of distribute_non_iid_v2

    """
    print("Poisoned workers:", poisoned_workers)
    num_workers = len(shuffled_data_shards)

    #Compile the shards of X & Y data into one flat X & Y data per worker
    compiled_data = dict()
    for worker in range(num_workers):
        compiled_data[worker] = dict()
        compiled_data[worker]['X'] = list()
        compiled_data[worker]['Y'] = list()
        for shard in shuffled_data_shards[worker]:
            #("shard:", shard)
            shard_x = shard[0].numpy()
            #shard_y = torch.unsqueeze(shard[1], 1).numpy()
            shard_y = shard[1].numpy()
            #print("shard_x:", type(shard_x), shard_x.shape)
            #print("shard_y:", type(shard_y), shard_y.shape)
            for x_elem in shard_x:
                #print("x_elem type: {}. x_elem shape: {}".format(type(x_elem), x_elem.shape))
                compiled_data[worker]['X'].append(x_elem)
            #for y_elem in torch.unsqueeze(shard[1], 0).numpy():
            for y_elem in shard_y:
                #print("y_elem type: {}. y_elem shape: {}".format(type(y_elem), y_elem.shape))
                compiled_data[worker]['Y'].append(y_elem)

        compiled_data[worker]['X'] = np.array(compiled_data[worker]['X'])
        compiled_data[worker]['Y'] = np.array(compiled_data[worker]['Y'])
        #print("Worker: {}. Size of X data: {}. Size of Y data: {}.".format(worker, compiled_data[worker]['X'].shape,
        #                                                                   compiled_data[worker]['Y'].shape))
        #print("Sample x:", compiled_data[worker]['X'][0].shape)
        #compiled_data[worker]['X'] = np.array(compiled_data[worker]['X'])
        #compiled_data[worker]['Y'] = np.array(compiled_data[worker]['Y'])

        if worker in poisoned_workers:
            dom_propn = np.sum(np.where(compiled_data[worker]['Y'] == poisoned_class, 1, 0))/len(compiled_data[worker]['Y'])
            target_propn = np.sum(np.where(compiled_data[worker]['Y'] == target_class, 1, 0))/len(compiled_data[worker]['Y'])
            print("Concealed class. Poisoned worker: {}. Poisoned class: {}. Propn: {}. Target class: {}. Propn: {}".format(worker, poisoned_class, dom_propn, target_class, target_propn))


        #Check whether poisoned class is still the dominant label
        if (worker in poisoned_workers) & (conceal_pois_class is False):
            #Here, the poisoned class is sending the correct class, thereby correct attribute skew
            compiled_data[worker]['Y'] = np.where(compiled_data[worker]['Y'] == poisoned_class, target_class, compiled_data[worker]['Y'])
            dom_propn = np.sum(np.where(compiled_data[worker]['Y'] == target_class, 1, 0))/len(compiled_data[worker]['Y'])
            print("Send true class. Poisoned worker: {}. Target class: {}. Propn: {}".format(worker, target_class, dom_propn))

    return compiled_data

def convert_to_dataloader(compiled_dataset, batch_size):
    #compiled_dataset - output of dd.compile_data: dictionary of each worker and its data X and Y, stored as dict
    X_comp = list()
    Y_comp = list()

    XY_data = compiled_dataset.values()
    for worker in XY_data:
        X_comp.extend(worker['X'])
        Y_comp.extend(worker['Y'])
    X_comp = np.squeeze(np.array(X_comp))
    Y_comp = np.squeeze(np.array(Y_comp))

    #print("X_comp", X_comp.shape)

    X_comp = X_comp.reshape((X_comp.shape[0], 1, X_comp.shape[1], X_comp.shape[2]))

    X_torch = torch.from_numpy(X_comp).float()
    Y_torch = torch.from_numpy(Y_comp).float()
    print("X data size:", X_torch.size())
    dataset = TensorDataset(X_torch, Y_torch)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    return dataloader