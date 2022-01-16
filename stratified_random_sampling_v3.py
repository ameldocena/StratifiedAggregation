from utilities import SelectionStrategy
import data_distribution as dd
import numpy as np
import random as rd
import statistics as stat
import math
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering

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
        self.stratification1 = KWARGS['stratify1'] #First stage stratification
        self.stratification2 = KWARGS['stratify2'] #Second stage stratification
        self.stratify_kwargs_s1 = KWARGS['stratify_kwargs_s1'] #kwargs for unsupervised stratification stage 1
        self.stratify_kwargs_s2 = KWARGS['stratify_kwargs_s2'] #kwargs for unsupervised stratification stage 1

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

    def count_total_qty(self):
        # Counts the total quantity of data per worker
        quantity = dict()
        workers = self.data_disbn.keys()
        for worker in workers:
            total = len(self.data_disbn[worker]['Y'])
            quantity[worker] = total

        return quantity

    def propn_labels_perworker(self):
        """
        Proportion of labels per worker.
        """
        propn_labels = dict()
        workers = self.data_disbn.keys()
        labels_qty = self.count_labels_qty()
        total_qty = self.count_total_qty()
        for worker in workers:
            propn_labels[worker] = list()
            for label in range(self.nlabels):
                propn_labels[worker].append(labels_qty[worker][label] / total_qty[worker])

        return propn_labels

    ##Score to measure label skew
    def index_skewed_label(self):
        """
        Returns the index of label a worker's data is skewed toward the most
        :return:
        """
        propn_labels = self.propn_labels_perworker()

        skewed_labels = list() #container for index of skewed label

        for worker in propn_labels.keys():
            skewed_label = propn_labels[worker].index(max(propn_labels[worker]))
            skewed_labels.append(skewed_label)
        return skewed_labels

    def unsupervised_stratify(self, technique, scores, workers, kwargs):
        """
        Inputs:
        technique - scalar (str): unsupervised clustering technique
        scores - dict: contains different scores/features for unsupervised clustering; wherein, each score is a dict of workers (keys)
        and the values are the domain for that score
        workers - worker ids
        kwargs - dict: arguments for the unsupervised technique

        Output:
        strata - dictionary: keys are the strata, values are the workers
        """

        # Step 1. We stack all scores
        scores_all = list()
        for score in scores.keys():
            score_values = list(scores[score].values())
            scores_all.append(score_values)
        scores_all = np.array(scores_all).reshape(len(workers), -1)
        # print("Scores_all:", scores_all.shape)
        if len(scores_all.shape) == 1:
            scores_all = scores_all.reshape(-1, 1)
        print("Scores all shape:", scores_all.shape)
        # Step 2. Stratification
        if technique == 'kmeans':
            clusters = KMeans(n_clusters=kwargs['n_clusters'], max_iter=kwargs['max_iters'],
                              random_state=kwargs['seed']).fit(scores_all)
            # print("KMeans:", kmeans.labels_)
        elif technique == 'spectral':
            clusters = SpectralClustering(n_clusters=kwargs['n_clusters'],
                                          n_neighbors=kwargs['n_neighbors'],
                                          affinity=kwargs['affinity'],
                                          random_state=kwargs['random_state']).fit(scores_all)
        else:
            raise("Please retry unsupervised clustering technique")

        #print("Results:", clusters.labels_)

        # Container for the stratified workers. Each stratum is a list
        strata = dict()

        # Need: Retrieve the worker clustered by the score.
        ##We append the respective workers in each stratum
        for stratum in range(kwargs['n_clusters']):
            strata['strata' + str(stratum + 1)] = list()

            clustered = np.argwhere(clusters.labels_ == stratum).reshape(-1, ).tolist()
            #print("Stratum: {}. Clustered: {}".format(stratum, clustered))
            for index in clustered:
                strata['strata' + str(stratum + 1)].append(workers[index])

        return strata

    def take_workers_greedily(self, strata, nsamples, ceiling = False):
        """
        Selects workers from strata pool greedily.
        Greedy selection of samples
        1. Sort the strata by elements in increasing order
        2. Take samples from each stratum either the entire stratum or the number of samples to take,
            whichever is the minimum.
        3. We take the remaining samples-to-take from the last ordered stratum
        :param strata:
        :return: selected_workers_list - selected workers stored as a list
            stratified_workers_dict - selected workers per stratum stored as a dict, wherein the keys are stratum
            ceiling - whether we use math.ceiling (math.floor) in determining the take_samples
        """

        #Number of non-empty stratum
        nstrata = 0
        for stratum in strata.keys():
            if len(strata[stratum]) > 0:
                nstrata += 1
        #print("Non-empty strata:", nstrata)
        # nstrata = sum(strata.keys)
        # nstrata = len(list(strata.keys())) #number of strata
        if ceiling is True:
            take_samples = math.ceil(nsamples/ nstrata)
        else:
            take_samples = math.floor(nsamples/nstrata)
        stratified_workers_dict = dict()
        selected_workers_list = list()

        #We determine the samples to take depending on the remaining number of samples to take
        if math.floor(nsamples/nstrata) > 0: #If we are sure that the elements in each stratum is enough for take_samples
            samples2take = take_samples * nstrata
        else:
            samples2take = nsamples

        print("Total samples to take: {}. Samples to take per stratum {}.".format(samples2take, take_samples))

        sorted_strata = sorted(strata.keys(), key=lambda x: len(strata[x]))
        #print("Sorted strata:", sorted_strata)

        for stratum in sorted_strata:
            if samples2take > 0:
                print("Samples to take:", samples2take)
                print("Stratum:", stratum, strata[stratum], len(strata[stratum]))
                # If number of samples to be taken is greater or equal to the number of
                # workers to choose from in given stratum, then take all members of that stratum
                min_take_samples = min(len(strata[stratum]), take_samples)
                select_stratum = rd.sample(strata[stratum], min_take_samples)
                print("Take samples:", min_take_samples, select_stratum)
                for worker in select_stratum:
                    selected_workers_list.append(worker)
                stratified_workers_dict[stratum] = select_stratum
                samples2take -= take_samples
            else:
                stratified_workers_dict[stratum] = []

        print("selected workers list:", selected_workers_list)
        print("stratified workers dict:", stratified_workers_dict)

        return selected_workers_list, stratified_workers_dict

    def greedy_selection(self, strata):
        """
        Greedy selection of samples.
        :param strata:
        :return:
        """
        # Step 1.
        print("Step 1")
        s1 = self.take_workers_greedily(strata, self.nsample)
        remainder = self.nsample - len(s1[0]) #There will be remainder. We loop over the take_workers_greedily once more.
        s1_strata_dict = s1[1].copy()
        remainder_strata = dict()
        for stratum in s1_strata_dict:
            print("strata: {}. s1: {}".format(strata[stratum], s1_strata_dict[stratum]))
            remainder_strata[stratum] = list(set(strata[stratum]) - set(s1_strata_dict[stratum]))
        print("Remainder strata:", remainder_strata)

        # Step 2. Deal with the remainder.
        print("Step 2")
        s2 = self.take_workers_greedily(remainder_strata, remainder, ceiling = True)

        # Step 3. Combine (or extend) s1 and s2
        print("Step 3")
        selected_workers_list = s1[0].copy()
        stratified_workers_dict = s1[1].copy()
        selected_workers_list.extend(s2[0]) #Extend the list
        for stratum in s1_strata_dict: #Extend the dict
            stratified_workers_dict[stratum].extend(s2[1][stratum])
        print("selected workers list:", selected_workers_list)
        print("stratified workers dict:", stratified_workers_dict)
        return selected_workers_list, stratified_workers_dict

    def influence_stratum(self, strata, poisoned_workers, influence):
        """
        Byzantine workers can influence to be in a certain (or specified) stratum
        :param influence:
        :return: strata
        """

        # Poisoned workers ability to influence to be in one stratum
        ##Params: switch, target stratum to influence
        print("Strata prior:", strata)
        if influence != None:
            # Step 1. We remove each poisoned worker from their initial (or benign) stratum assignment
            # Step 2.Then assign them to their target stratum

            # Step 1
            poisoned_copy = poisoned_workers.copy()
            for stratum in strata.keys():
                print("Prior stratum:", strata[stratum])
                for pois in poisoned_copy:
                    if pois in strata[stratum]:
                        print("Found poisoned worker:", pois)
                        strata[stratum].remove(pois)
                        # poisoned_copy.remove(pois)
                print("After stratum:", strata[stratum])
                print("Pois_copy left:", poisoned_copy)

            # Step 2
            strata[influence].extend(poisoned_workers)

            print("Influenced strata:", strata)
            print("Poisoned stratum:", poisoned_workers, strata[influence])

        return strata

    """
    Stage 1: Stratify based on non-IID attributes
    #We ask for what kind of stratification here:

    I MAY HAVE TO ASK FOR KWARGS FOR THE UNSUPERVISED STRATIFICATION PARAMS
    """

    def select_round_workers(self, root_disbn, poisoned_workers, workers, influence=None, worker_disbn = None):
        """
        :param root_disbn: root distribution
        :param poisoned_workers:
        :param workers: list of workers to select from
        :param influence: str - the stratum where poisoned workers would want to be stratified to
        :param worker_disbn: dict - list of 'orig', 'skewed', and 'poisoned' workers
        :return: stratified workers
        """

        non_iid_scores = dict()
        non_iid_scores['score1'] = self.propn_labels_perworker() #Stratify based on non-IID attributes

        if self.stratification1=="kmeans":
            print("Stage 1: KMeans stratification is selected")
            strata = self.unsupervised_stratify('kmeans', non_iid_scores, workers, self.stratify_kwargs_s1) #nclusters here should be kwargs
        elif self.stratification1=="spectral":
            print("Stage 1: Spectral stratification is selected")
            strata = self.unsupervised_stratify('spectral', non_iid_scores, workers, self.stratify_kwargs_s1)

        #Poisoned workers ability to influence to be in one stratum
        ##Params: switch, target stratum to influence
        strata = self.influence_stratum(strata, poisoned_workers, influence)

        #Greedy selection of workers
        selected_workers_list, stratified_workers_dict = self.greedy_selection(strata)

        return selected_workers_list, stratified_workers_dict

    """
    STAGE 2: UNSUPERVISED STRATIFY TO FILTER OUT POTENTIALLY BAD UPDATES
        > nclusters
        > select centroids above threshold (potentially > 0.50)
        
        Here we shall stratify selected_workers_list by features used to stratify potentially bad updates
        This means then that the scores (or feature space of the parameter) can be 
    """
    def compute_fscores(self, local_clients_results):
        """
        Computes the F-scores of updates given test result performance
        :param test_results: dict of class precision and recall
        :return:
        fscore - dict: keys (worker), values: (np.array) fscores for all labels
        """

        # performance = dict()
        # loss = dict()
        precision = dict()
        recall = dict()
        fscore = dict()

        for worker in local_clients_results:
            #performance[worker] = local_clients_results['epoch_' + str(epoch)][worker][0]
            #loss['epoch_' + str(epoch)][worker] = local_clients_results['epoch_' + str(epoch)][worker][1]
            precision[worker] = local_clients_results[worker][2]
            recall[worker] = local_clients_results[worker][3]

            # Compute F-score
            prec = precision[worker]
            rec = recall[worker]
            fscore[worker] = 2 * prec * rec / (prec + rec)

            # We are replacing an NaN f-score as 0.0 for clustering purposes.
            # NaN will not be useful in clustering.
            fscore[worker] = np.nan_to_num(fscore[worker]) #default replacement is 0.0

        return fscore

    """
    Compute meaningful class f-score:
    1. F-score of skewed label
    2. Weighted average of F-score
    """
    ##Score to measure F-score of skewed label
    def fscore_skewedlabel(self, fscores):
        """
        Compute the F-score of the skewed label
        :param fscores:
        :return:
        """
        skewed_labels = self.index_skewed_label()
        print("skewed labels:", skewed_labels)
        fscores_skewed = dict()
        for worker in fscores.keys():
            fscores_skewed[worker] = fscores[worker][skewed_labels[worker]]

        return fscores_skewed

    ##Score to compute average F-score weighted by the label propn
    def average_fscore(self, fscores):
        """
        Computes the average fscore weighted by the label propn
        :param fscores:
        :return:
        """
        propn_labels = self.propn_labels_perworker()
        ave_fscores = dict()

        for worker in fscores.keys():
            ave_fscores[worker] = np.array(propn_labels[worker]) * fscores[worker]

        return ave_fscores

    ##Worker precision and recall of skewed label
    def precision_skewedlabel(self, local_clients_results):
        skewed_labels = self.index_skewed_label()
        print("skewed labels:", skewed_labels)

        precision_skewed = dict()
        for worker in local_clients_results.keys():
            precision = np.nan_to_num(local_clients_results[worker][2])
            precision_skewed[worker] = precision[skewed_labels[worker]]

        return precision_skewed

    def recall_skewedlabel(self, local_clients_results):
        skewed_labels = self.index_skewed_label()
        print("skewed labels:", skewed_labels)

        recall_skewed = dict()
        for worker in local_clients_results.keys():
            recall = np.nan_to_num(local_clients_results[worker][3])
            recall_skewed[worker] = recall[skewed_labels[worker]]

        return recall_skewed

    def filter_centroids(self, clusters, scores, threshold):
        """
        Keeps clusters whose centroids pass the threshold
        S1. Compute the centroid score of each cluster
        S2. Keep centroid that pass the threshold
        :param clusters:
        :param score:
        :return:
        """
        keep_clusters = clusters.copy()
        centroids = dict()
        print("Clusters:", clusters)
        print("Scores:", scores)

        for cluster in clusters: #TO-DO: Shouldn't this be at the worker level? We compute (match) the score at the worker level
            sum_score = list()
            for worker in clusters[cluster]:
                sum_score.append(scores[worker]) #This code assumes there is one score per worker
            average = sum(sum_score)/len(sum_score) #Compute the scores
            centroids[cluster] = average
            if average < threshold:
                keep_clusters.pop(cluster)
        print("Centroids:", centroids)
        return keep_clusters

    def filter_centroids_v2(self, clusters, scores, threshold):
        """
        Keeps clusters whose centroids pass the threshold.
        Here the scores are multi-dimensional.

        S1. We compute the centroid precision and recall
        S2. We then compute its F-score
        S3. We filter out those clusters that do not pass the threshold

        :param clusters:
        :param score:
        :return:
        """
        keep_clusters = clusters.copy()
        centroids = dict()
        print("Clusters:", clusters)
        print("Scores:", scores)

        #S1. Compute centroid precision and recall
        for cluster in clusters:  # TO-DO: Shouldn't this be at the worker level? We compute (match) the score at the worker level
            list_prec = list()
            list_rec = list()
            for worker in clusters[cluster]:
                list_prec.append(scores['precision'][worker])  # This code assumes there is one score per worker
                list_rec.append(scores['recall'][worker])
            average_prec = sum(list_prec) / len(list_prec)  # Compute the scores
            average_rec = sum(list_rec) / len(list_rec)

            centroids[cluster] = 2 * average_prec * average_rec / (average_prec + average_rec)
            if centroids[cluster] < threshold:
                keep_clusters.pop(cluster)

        print("Centroids:", centroids)
        return keep_clusters

    def stratify_stage2(self, feature_space, workers, threshold):
        """
        S1. Clusters workers to stratify potentially bad updates based on the feature space.
        Here we use clustering techniques that allows specification for the number of clusters.
        S2. We compute the centroid of each cluster, and then keep only those that pass this threshold.
        (Implementation-wise, we can do the inverse.)

        :param : The feature on which to stratify the workers.
        :return:
        """
        print("Centroid threshold:", threshold)

        stage2_scores = dict()
        stage2_scores['score'] = feature_space

        #S1: Stratify
        if self.stratification2=="kmeans":
            print("Stage 2: KMeans stratification is selected")
            strata_stage2 = self.unsupervised_stratify('kmeans', stage2_scores, workers, self.stratify_kwargs_s2) #nclusters here should be kwargs
        elif self.stratification2=="spectral":
            print("Stage 2: Spectral stratification is selected")
            strata_stage2 = self.unsupervised_stratify('spectral', stage2_scores, workers, self.stratify_kwargs_s2)

        #Convert the

        #S2: Threshold condition
        #a. Compute centroid for each cluster
        #b. Keep clusters whose centroids pass the threshold
        po_stage2 = self.filter_centroids(strata_stage2, feature_space, threshold) #TO-EDIT

        #Return as list
        po_stage2_list = list()
        for stratum in po_stage2:
            po_stage2_list.extend(po_stage2[stratum])

        return po_stage2_list

    def stratify_stage2_v2(self, feature_space, workers, threshold):
        """
        S1. Clusters workers to stratify potentially bad updates based on the feature space.
        Here we use clustering techniques that allows specification for the number of clusters.
        S2. We compute the centroid of each cluster, and then keep only those that pass this threshold.
        (Implementation-wise, we can do the inverse.)

        :param : The feature on which to stratify the workers.
        :return:
        """
        print("Centroid threshold:", threshold)

        # stage2_scores = dict()
        # stage2_scores['score'] = feature_space

        #S1: Stratify
        if self.stratification2=="kmeans":
            print("Stage 2: KMeans stratification is selected")
            strata_stage2 = self.unsupervised_stratify('kmeans', feature_space, workers, self.stratify_kwargs_s2) #nclusters here should be kwargs
        elif self.stratification2=="spectral":
            print("Stage 2: Spectral stratification is selected")
            strata_stage2 = self.unsupervised_stratify('spectral', feature_space, workers, self.stratify_kwargs_s2)

        #Convert the

        #S2: Threshold condition
        #a. Compute centroid for each cluster
        #b. Keep clusters whose centroids pass the threshold
        po_stage2 = self.filter_centroids_v2(strata_stage2, feature_space, threshold) #TO-EDIT

        #Return as list
        po_stage2_list = list()
        for stratum in po_stage2:
            po_stage2_list.extend(po_stage2[stratum])

        return po_stage2_list
    """
    Here we do an if condition to filter out bad clusters
    We scrap out clusters whose centroids are below a set threshold
    Operation: S1. Given the F-scores, we compute the average F-score of a given cluster
    #S2. We keep only the cluster that passes this threshold
    """

    """
    STAGE 3: FILTER OUT BAD UPDATES FROM STAGE 1
    S1. We pair Stage 1 container with in Stage 2 container, probably do a for-loop. We do a set difference
    where Stage 1 is the minuend and Stage 2 is the subtrahend.
    S2. We return as a list selected_workers_list, and a dict, stratified_workers_dict 
    """
    def select_round_workers_stage2(self, strata_stage1, strata_stage2):
        """
        Intersection between strata_stage1 and strata_stage2
        :param strata_stage1:
        :param strata_stage2: list
        :return:
        """
        selected_workers_list = list()
        stratified_workers_dict = dict()
        for stratum in strata_stage1:
            intersection = list(set(strata_stage1[stratum]).intersection(set(strata_stage2)))
            selected_workers_list.extend(intersection)
            stratified_workers_dict[stratum] = intersection

        return selected_workers_list, stratified_workers_dict

    """
    PO: RESAMPLE STRATUM
    In this case, no need since we are stratifying workers
    """