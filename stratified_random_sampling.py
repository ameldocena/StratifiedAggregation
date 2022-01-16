from utilities import SelectionStrategy
import data_distribution as dd
import numpy as np
import random as rd
import statistics as stat
import math
from sklearn.cluster import KMeans, DBSCAN

class StratifiedRandomSampling(SelectionStrategy):
    #PO1: We use an init
    #PO2: We just functions
    def __init__(self, data_disbn, KWARGS):
        """
        data_disbn - dict: workers as keys, X and Y as values

        """
        self.num_workers = len(data_disbn.keys()) #We can just use the keys of data_disbn
        self.data_disbn = data_disbn #workers X and Y data
        #self.root_disbn = root_disbn #for the attribute skew, potential solution #This could be added as function parameter
        self.nsample = KWARGS['NUM_WORKERS_PER_ROUND'] #Number of samples to be taken #Can be a function parameter
        #assert self.nsample % 2 == 0, " Number of samples to be taken should be divisible by 2"
        self.nlabels = KWARGS['nlabels'] #Number of labels in data
        self.stratification = KWARGS['stratify']
        self.nclusters = KWARGS['nclusters']

    def count_labels_qty(self):
        #Virtue shall shine through.
        #Counts the quantity of label per worker
        worker_quant_label = dict()
        workers = self.data_disbn.keys()
        for worker in workers:
            labels = self.data_disbn[worker]['Y']
            worker_quant_label[worker] = dict()
            #print("Worker: {}. Unique labels: {}".format(worker, np.unique(labels)))
            for label in range(self.nlabels):
                #print("worker: {}. label: {}.".format(worker, label))
                if label in labels:
                    label_count = len(np.extract(labels == label, labels))
                    worker_quant_label[worker][label] = label_count
        return worker_quant_label

    def label_skew(self):
        #Computes the label skew for each of the workers
        #Label skew is the number of labels (out of total) present in a workers training data
        label_skew_dict = dict()
        for worker in range(self.num_workers):
            labels = self.data_disbn[worker]['Y']
            #print("labels:", type(labels), labels.shape)
            label_skew_dict[worker] = len(np.unique(labels))/self.nlabels
            #print("length of label:", label_skew_dict[worker], np.unique(labels))

        return label_skew_dict

    def quanity_skew(self):
        #Computes quantity skew for each of the workers
        #Quantity skew is the quantity of worker data out of total data for a given label
        #labels per worker
        #total data for a given label

        workers = self.data_disbn.keys()
        total_quant_label = dict() #container for total quantity per label data
        for label in range(self.nlabels):
            total_quant_label[label] = 0

        #Compute the quantity per label for each worker
        worker_quant_label = self.count_labels_qty()

        #Compute the total quantity for each label
        for worker in workers:
            #print("Worker:", worker)
            unique_labels = np.unique(self.data_disbn[worker]['Y']) #unique labels in worker data
            for label in unique_labels:
                #print("Label:", label)
                #print("Worker quantity for given label:", worker_quant_label[worker][label])
                #print("Total quantity for given label:", total_quant_label[label])
                total_quant_label[label] += worker_quant_label[worker][label]
        print("worker_quant_label:", worker_quant_label)
        print("total_quant_label:", total_quant_label)
            #quant_skew_dict[worker][label] = worker_quant_label[worker][label]/total_quant_label[label]
            #print("Quantity skew:", quant_skew_dict[worker])

        #Compute the quantity skewness per label for each worker
        quant_skew_dict = dict()
        for worker in workers:
            unique_labels = np.unique(self.data_disbn[worker]['Y'])
            quant_skew_dict[worker] = dict()
            for label in unique_labels:
                quant_skew_dict[worker][label] = worker_quant_label[worker][label]/total_quant_label[label]
        print(quant_skew_dict)

        return quant_skew_dict
        #Remarks:
        #Eventually we would do the average. One possibility is to keep this structure but make an average.
        #Or, store as a list then measure the mean.

    #IDEA: We can stratify by the label and quantity skew.
    #We then add the attribute skew later.
    """
    Next methods:
    1. Set up the scores:
        > Cut 1: Label skewness and Average Quantity skewness
        > Cut 2: Attribute S and Label + Ave Quant S
    2. Stratify workers based on scores: 
        > Each stratum should be a list of workers
    3. Select workers: Should return a list
    
    PO:
    1. We can do Stratified FL under this first-cut setting. Advantage: We get to do a first-run!
    2. We can then just add as an additional score/refinement the attribute skew
    
    Remarks:
    1. On the number of training data. The label skewness somehow tapers off too much data.
    We can increase more training data for each label perhaps.
    
    On setting up the scores/stratification:
    We have four strata:
        a. We set up two scores and then take the median of both; this will be the threshold.
        b. We store as lists workers who belong within each stratum:
            Stratum 1 (upper-left Q): S1 >= median & S2 < median
            Stratum 2 (upper-right Q): S1 >= median & S2 >= median 
            Stratum 3 (lower-left): S1 < median & S2 < median
            Stratum 4 (lower-right): S1 < median & S2 >= median
        c. We then randomly pick workers from each of the stratum.
            Note that the proportion of selection can be proportional or preferential.
    
    The returned samples are what we feed in train_subset_clients() labeled as random_workers 
    """
    def feature_variation(self, root_disbn, train_data_pt):
        """
        Here we compute the variation between the root_disbn and the test_disbn.
        If we are setting the root_disbn to be the average training data, then we can use variance or standard deviation.

        Inputs:
        root_disbn - numpy: feature disbn; (here the average training data)
        test_disbn - numpy: training data disbn

        """

        std_dev = math.sqrt(np.var(root_disbn - train_data_pt))
        return std_dev

    def feature_variation(self, root_disbn):
        "Computes the average feature variation for each label per worker"


        std_dev = dict()  # dictionary of average standard deviation
        workers = self.data_disbn.keys()
        quant_data_worker = dict()  # quantity of worker data
        for worker in workers:
            # print(self.data_disbn[worker]['X'])
            quant_data_worker[worker] = len(self.data_disbn[worker]['X'])
        print("Quant worker data", quant_data_worker)

        for worker in workers:
            # Compute average standard deviation
            unique_labels = np.unique(self.data_disbn[worker]['Y'])  # unique labels per worker
            print("Worker: {}. Unique labels: {}".format(worker, unique_labels))
            segregated_label = dd.segregate_labels(self.data_disbn[worker])  # segregated data per label
            std_dev[worker] = dict()
            sqrd_diff = list()

            for label in unique_labels:
                n = len(segregated_label[label]['X'])
                for data_pt in segregated_label[label]['X']:
                    sqrd_diff.append(np.sum((data_pt - root_disbn[label]) ** 2))
                avg_sqrd_diff = np.sum(sqrd_diff) / n
                std_dev[worker][label] = np.sqrt(avg_sqrd_diff)

        return std_dev

    def attribute_skew(self, root_disbn):
        #Step 1. Measure the feature variation
        std_dev = self.feature_variation(root_disbn)

        #Step 2. Compute the propn of label data within workers data
        workers = self.data_disbn.keys()
        quant_label_worker = self.count_labels_qty()  # quantity of labels in worker data
        quant_data_worker = dict()  # quantity of worker data
        for worker in workers:
            # print(self.data_disbn[worker]['X'])
            quant_data_worker[worker] = len(self.data_disbn[worker]['X'])
        print("Quant worker data", quant_data_worker)
        label_propn_worker = dict()  # propn of each label in worker data

        print("Quant label workers:", quant_label_worker)
        print("Quant data worker:", quant_data_worker)
        for worker in workers:
            label_propn_worker[worker] = dict()
            unique_labels = np.unique(self.data_disbn[worker]['Y'])
            for label in unique_labels:
                label_propn_worker[worker][label] = quant_label_worker[worker][label]/quant_data_worker[worker]

        #Step 3. Compute average of feature variation weighted by label propn
        weighted_feature_variation = dict()
        for worker in workers:
            weighted_average = list() #average of feature variation weighted by the volume of label data
            unique_labels = np.unique(self.data_disbn[worker]['Y'])
            for label in unique_labels:
                weighted_average.append(label_propn_worker[worker][label]*std_dev[worker][label])
            weighted_feature_variation[worker] = sum(weighted_average)
        return weighted_feature_variation

    def score1(self, root_disbn):
        """
        Score 1: Attribute skewness weighted by the quantity skewness.
        Attribute skewness: We can take the average training data as root distribution
        For now, we compare the variation based on the features

        Inputs:
        root_disbn - dict: dictionary of average training data for each label
        """
        #Step 1. Measure attribute skewness
        attribute_skewness = self.attribute_skew(root_disbn)
        print("Attr skewness:", attribute_skewness)
        #Step 2. Compute label skewness
        label_skewness = self.label_skew()
        #print("Quant skewness:", quant_skewness)

        #Step 3. Score is directly proportional to the average feature variation but
        #inversely proportional to the label skewness
        score = dict()
        workers = self.data_disbn.keys()
        for worker in workers:
            #score[worker] = attribute_skewness[worker]/label_skewness[worker]
            score[worker] = attribute_skewness[worker]
        return score

    def score2(self):
        """
        Score 2: Average quantity skewness
        quant_skew - workers as keys, and labels as values, wherein each value/label is a key
        to the proportion of quantity relative to the total volume.

        quant_skew.keys() - workers
        quant_skew.values() - dict: label (as keys) and relative volume (as values)
        quant_skew.values().keys() - labels
        quant_skew.values().values - relative volume
        """
        quant_skew = self.quanity_skew()
        workers = quant_skew.keys()
        score = dict()
        for worker in workers:
            quant_skew_labels = list(quant_skew[worker].values())
            score[worker] = sum(quant_skew_labels)/len(quant_skew_labels)

        #print("Score:", score)
        return score

    def score3(self):
        label_skew = self.label_skew()
        return label_skew

    def stratify(self, scores):
        #Stratify the workers
        #Compute the threshold; here the median score
        #If worker's score is below the threshold, we group into one stratum
        #If beyond, we group in another stratum
        #scores - scores to stratify with; can be a tuple/list/dictionary

        # scores1 = scores['score1']
        # scores2 = scores['score2']
        # median_score1 = stat.median(list(scores1.values())) #Assumes one value setting for scores
        # median_score2 = stat.median(list(scores2.values()))  # Assumes one value setting for scores

        median_scores = dict()
        for score in scores.keys():
            median_scores[score] = stat.median(list(scores[score].values()))
            print("Median score:", score, median_scores[score])

        nstrata = 4
        strata = dict()
        for stratum in range(nstrata):
            strata['strata' + str(stratum+1)] = list()

        scores1 = scores['score1']
        scores2 = scores['score2']
        scores3 = scores['score3']

        median_score1 = median_scores['score1'] #attribute skew
        median_score2 = median_scores['score2'] #quantity skew
        median_score3 = median_scores['score3'] #label skew

        # for worker in range(self.num_workers):
        #     print("worker: {}. score1: {}. score2: {}".format(worker, scores1[worker], scores2[worker], scores3[worker]))
            # if (scores1[worker] < median_score1) & (scores2[worker] < median_score2) & (scores3[worker] < median_score3):
            #     strata['strata1'].append(worker)
            # elif (scores1[worker] < median_score1) & (scores2[worker] < median_score2) & (scores3[worker] >= median_score3):
            #     strata['strata2'].append(worker)
            # elif (scores1[worker] < median_score1) & (scores2[worker] >= median_score2) & (scores3[worker] < median_score3):
            #     strata['strata3'].append(worker)
            # if (scores1[worker] < median_score1) & (scores2[worker] >= median_score2) & (scores3[worker] >= median_score3):
            #     strata['strata4'].append(worker)
            # elif (scores1[worker] >= median_score1) & (scores2[worker] < median_score2) & (scores3[worker] < median_score3):
            #     strata['strata5'].append(worker)
            # elif (scores1[worker] >= median_score1) & (scores2[worker] < median_score2) & (scores3[worker] >= median_score3):
            #     strata['strata6'].append(worker)
            # elif (scores1[worker] >= median_score1) & (scores2[worker] >= median_score2) & (scores3[worker] < median_score3):
            #     strata['strata7'].append(worker)
            # elif (scores1[worker] >= median_score1) & (scores2[worker] >= median_score2) & (scores3[worker] >= median_score3):
            #     strata['strata8'].append(worker)

        # for worker in range(self.num_workers):
        #     print("worker: {}. score1: {}. score2: {}".format(worker, scores1[worker], scores2[worker]))
        #     if (scores1[worker] < median_score1) & (scores2[worker] >= median_score2):
        #         strata['strata1'].append(worker)
        #     elif (scores1[worker] >= median_score1) & (scores2[worker] >= median_score2):
        #         strata['strata2'].append(worker)
        #     elif (scores1[worker] < median_score1) & (scores2[worker] < median_score2):
        #         strata['strata3'].append(worker)
        #     elif (scores1[worker] >= median_score1) & (scores2[worker] < median_score2):
        #         strata['strata4'].append(worker)

        for worker in range(self.num_workers):
            #print("worker: {}. score1: {}. score2: {}".format(worker, scores['scores1'][worker], scores['scores2'][worker]))
            if (scores['score1'][worker] < median_scores['score1']) & (scores['score2'][worker] >= median_scores['score2']):
                strata['strata1'].append(worker)
            elif (scores['score1'][worker] >= median_scores['score1']) & (scores['score2'][worker] >= median_scores['score2']):
                strata['strata2'].append(worker)
            elif (scores['score1'][worker] < median_scores['score1']) & (scores['score2'][worker] < median_scores['score2']):
                strata['strata3'].append(worker)
            elif (scores['score1'][worker] >= median_scores['score1']) & (scores['score2'][worker] < median_scores['score2']):
                strata['strata4'].append(worker)

        # for worker in range(self.num_workers):
        #     if scores1[worker] < median_score1:
        #         strata['strata1'].append(worker)
        #     elif scores1[worker] >= median_score1:
        #         strata['strata2'].append(worker)
        # for worker in range(self.num_workers):
        #     if scores2[worker] < median_score2:
        #         strata['strata3'].append(worker)
        #     elif scores2[worker] >= median_score2:
        #         strata['strata4'].append(worker)
        return strata

    def unsupervised_stratify(self, technique, scores, nclusters):
        """
        Output:
        strata - dictionary: keys are the strata, values are the workers
        """

        # Step 1. We stack all scores
        scores_all = list()
        for score in scores.keys():
            score_values = list(scores[score].values())
            scores_all.append(score_values)
        scores_all = np.array(scores_all).transpose()
        print("Scores_all:", scores_all.shape, scores_all)

        if technique=='kmeans':
            cluster = KMeans(nclusters, random_state=1234).fit(scores_all)
            print("KMeans:", cluster.labels_)
        elif technique=='dbscan':
            cluster = DBSCAN(eps = 0.5, min_samples = 5).fit(scores_all)
            print("DBSCAN:", cluster.labels_)

        # Container for the stratified workers. Each stratum is a list
        strata = dict()

        ##We append the respective workers in each stratum
        for stratum in range(nclusters):
            strata['strata' + str(stratum + 1)] = list()
            for worker in range(scores_all.shape[0]):
                if cluster.labels_[worker] == stratum:
                    strata['strata' + str(stratum + 1)].append(worker)

        return strata

    def kmeans_stratify(self, scores, nclusters):
        """
        Output:
        strata - dictionary: keys are the strata, values are the workers
        """

        # Step 1. We stack all scores
        scores_all = list()
        for score in scores.keys():
            score_values = list(scores[score].values())
            scores_all.append(score_values)
        scores_all = np.array(scores_all).transpose()
        print("Scores_all:", scores_all.shape, scores_all)

        kmeans = KMeans(nclusters, random_state=1234).fit(scores_all)
        print("KMeans:", kmeans.labels_)
        # Container for the stratified workers. Each stratum is a list
        strata = dict()

        ##We append the respective workers in each stratum
        for stratum in range(nclusters):
            strata['strata' + str(stratum + 1)] = list()
            for worker in range(scores_all.shape[0]):
                if kmeans.labels_[worker] == stratum:
                    strata['strata' + str(stratum + 1)].append(worker)

        return strata

    def select_round_workers(self, root_disbn, poisoned_workers, influence=None):
        """
        :param root_disbn: root distribution
        :param poisoned_workers:
        :param influence: str - the stratum where poisoned workers would want to be stratified to
        :return: stratified workers
        """

        #take_samples can be a scalar for uniform proportion of selection
        #Or, it can be a dict/list corresponding to the proportion to take from each of the
        #stratum.

        #WE CAN MAKE take_samples as a function:
        #1. Equal propotion: nsamples/nstrata
        #2. Preferential where provided by the user

        scores = dict()
        scores['score1'] = self.score1(root_disbn) #attribute skew
        scores['score2'] = self.score2() #quantity skew
        scores['score3'] = self.score3() #label skew
        # scores['score2'] = self.score3() #label skew
        # scores['score3'] = self.score2() #quant skew

        #We ask for what kind of stratification here:
        if self.stratification=='median':
            print("Median stratification is selected")
            strata = self.stratify(scores)
        elif self.stratification=="kmeans":
            print("KMeans stratification is selected")
            strata = self.unsupervised_stratify(scores, 'kmeans', self.nclusters)
        elif self.stratification=="dbscan":
            print("DBSCAN stratification is selected")
            strata = self.unsupervised_stratify(scores, 'dbscan', self.nclusters)

        #Poisoned workers ability to influence to be in one stratum
        ##Params: switch, target stratum to influence
        print("Strata prior:", strata)
        if influence != None:
            #Step 1. We remove each poisoned worker from their initial (or benign) stratum assignment
            #Step 2.Then assign them to their target stratum

            #Step 1
            poisoned_copy = poisoned_workers.copy()
            for stratum in strata.keys():
                print("Prior stratum:", strata[stratum])
                for pois in poisoned_copy:
                    if pois in strata[stratum]:
                        print("Found poisoned worker:", pois)
                        strata[stratum].remove(pois)
                        #poisoned_copy.remove(pois)
                print("After stratum:", strata[stratum])
                print("Pois_copy left:", poisoned_copy)

            #Step 2
            strata[influence].extend(poisoned_workers)

            print("Influenced strata:", strata)
            print("Poisoned stratum:", poisoned_workers, strata[influence])


        #Number of non-empty stratum
        nstrata = 0
        for stratum in strata.keys():
            if len(strata[stratum]) > 0:
                nstrata += 1
        print("Non-empty strata:", nstrata)
        # nstrata = sum(strata.keys)
        # nstrata = len(list(strata.keys())) #number of strata
        take_samples = math.ceil(self.nsample/nstrata)

        stratified_workers_dict = dict()
        selected_workers_list = list()
        samples2take = self.nsample

        """
        Greedy selection of samples
        1. Sort the strata by elements in increasing order
        2. Take samples from each stratum either the entire stratum or the number of samples to take,
            whichever is the minimum.
        3. We take the remaining samples-to-take from the last ordered stratum
        
        We can turn this into a function of its own.
        """
        sorted_strata = sorted(strata.keys(), key = lambda x: len(strata[x]))
        print("Sorted strata:", sorted_strata)
        for stratum in sorted_strata[ : nstrata-1]:
            print("Stratum:", stratum, strata[stratum], len(strata[stratum]))
            # If number of samples to be taken is greater or equal to the number of
            # workers to choose from in given stratum, then take all members of that stratum
            min_take_samples = min(len(strata[stratum]), take_samples)
            select_stratum = rd.sample(strata[stratum], min_take_samples)
            print("Take samples:", min_take_samples, select_stratum)
            for worker in select_stratum:
                selected_workers_list.append(worker)
            stratified_workers_dict[stratum] = select_stratum
            samples2take -= len(select_stratum)
        # We pick the remaining samples to take from the last stratum
        stratum = sorted_strata[nstrata-1]
        print("Stratum:", stratum, strata[stratum], len(strata[stratum]))
        select_stratum = rd.sample(strata[stratum], samples2take)
        print("Take samples:", samples2take, select_stratum)
        for worker in select_stratum:
            selected_workers_list.append(worker)
        stratified_workers_dict[stratum] = select_stratum

        print("Selected workers:", len(selected_workers_list), selected_workers_list)
        return selected_workers_list, stratified_workers_dict