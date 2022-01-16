from data_distribution import replace_N_with_M
from utilities import RandomSelectionStrategy
from saving_utils import get_class_label_from_num
import random as rd
import numpy as np
from server_core_v3 import run_exp
import torch

"""
450 - FedAvg-Clean
550 - FedAvg-Pois
650 - SRS-Krum Done
610 - SRS-Tmean*
750 - StRS-StrKrum: 0.50, 0.50 Done
850 - StRS-StrMed: 0.50, 0.50 Upnext
950 - StRS-StrTMean*
"""
strata_weights_unif = [{'strata1': 0.10, 'strata2': 0.10, 'strata3' : 0.10, 'strata4': 0.10, 'strata5': 0.10,
     'strata6': 0.10, 'strata7': 0.10, 'strata8' : 0.10, 'strata9': 0.10, 'strata10': 0.10}]

strata_weights_list_dyn = [
    {'strata1': 0.10, 'strata2': 0.10, 'strata3' : 0.10, 'strata4': 0.10, 'strata5': 0.10,
     'strata6': 0.10, 'strata7': 0.10, 'strata8' : 0.10, 'strata9': 0.10, 'strata10': 0.10},

    {'strata1': 0.20, 'strata2': 0.0389, 'strata3' : 0.0389, 'strata4': 0.0389, 'strata5': 0.0389,
     'strata6': 0.0389, 'strata7': 0.0389, 'strata8' : 0.0389, 'strata9': 0.0389, 'strata10': 0.0389},

    {'strata1': 0.0389, 'strata2': 0.20, 'strata3': 0.0389, 'strata4': 0.0389, 'strata5': 0.0389,
     'strata6': 0.0389, 'strata7': 0.0389, 'strata8': 0.0389, 'strata9': 0.0389, 'strata10': 0.0389},

    {'strata1': 0.0389, 'strata2': 0.0389, 'strata3': 0.20, 'strata4': 0.0389, 'strata5': 0.0389,
     'strata6': 0.0389, 'strata7': 0.0389, 'strata8': 0.0389, 'strata9': 0.0389, 'strata10': 0.0389},

    {'strata1': 0.0389, 'strata2': 0.0389, 'strata3': 0.0389, 'strata4': 0.20, 'strata5': 0.0389,
     'strata6': 0.0389, 'strata7': 0.0389, 'strata8': 0.0389, 'strata9': 0.0389, 'strata10': 0.0389},

    {'strata1': 0.0389, 'strata2': 0.0389, 'strata3' : 0.0389, 'strata4': 0.0389, 'strata5': 0.20,
     'strata6': 0.0389, 'strata7': 0.0389, 'strata8' : 0.0389, 'strata9': 0.0389, 'strata10': 0.0389},

    {'strata1': 0.0389, 'strata2': 0.0389, 'strata3' : 0.0389, 'strata4': 0.0389, 'strata5': 0.0389,
     'strata6': 0.20, 'strata7': 0.0389, 'strata8' : 0.0389, 'strata9': 0.0389, 'strata10': 0.0389},

    {'strata1': 0.0389, 'strata2': 0.0389, 'strata3': 0.0389, 'strata4': 0.0389, 'strata5': 0.0389,
     'strata6': 0.0389, 'strata7': 0.20, 'strata8': 0.0389, 'strata9': 0.0389, 'strata10': 0.0389},

    {'strata1': 0.0389, 'strata2': 0.0389, 'strata3': 0.0389, 'strata4': 0.0389, 'strata5': 0.0389,
     'strata6': 0.0389, 'strata7': 0.0389, 'strata8': 0.20, 'strata9': 0.0389, 'strata10': 0.0389},

    {'strata1': 0.0389, 'strata2': 0.0389, 'strata3': 0.0389, 'strata4': 0.0389, 'strata5': 0.0389,
     'strata6': 0.0389, 'strata7': 0.0389, 'strata8': 0.0389, 'strata9': 0.20, 'strata10': 0.0389},

    {'strata1': 0.0389, 'strata2': 0.0389, 'strata3': 0.0389, 'strata4': 0.0389, 'strata5': 0.0389,
     'strata6': 0.0389, 'strata7': 0.0389, 'strata8': 0.0389, 'strata9': 0.0389, 'strata10': 0.20}
]

#kwargs_kmeans = {'n_clusters' : n_clusters, 'max_iters' : max_iters, 'seed' : seed}
#kwargs_spectral = {'n_clusters' : n_clusters, 'n_neighbors' : n_neighbors, 'affinity' : , affinity, 'random_state' : random_state}

kw_srsFedAvg = {
        "NUM_WORKERS_PER_ROUND" : 50,
        "SELECTION_STRATEGY": "Simple_RS", #sample selection: Simple_RS or Stratified_RS
        "nlabels": 10,

        #Stratified kwargs
        "strata_weights": None, #stratum weights for weighted aggregation
        "conceal_pois_class": None, #Poisoned worker either conceals poisoned class
        "influence": None, #stratum to influence: "strata" + stratum number, default None
        "stratify1": None, #stratification technique: 'median', 'kmeans', 'spectral
        "stratify2": None,
        "stratify_kwargs_s1": None, #Kwargs for the unsupervised strat technique
        "stratify_kwargs_s2": None,
        "assumed_workers_per_round_stratum": None, #We divide NUM_WORKERS_PER_ROUND by the number of clusters
        "assumed_poisoned_per_round_stratum": None #Ratio of poisoned workers to num_workers applied to assumed workers per round
    }

kw_strsTMeanUnif = {
        "NUM_WORKERS_PER_ROUND" : 50, #4, #25
        "ASSUMED_POISONED_WORKERS_PER_ROUND" : 5, #an estimate of how many of the selected workers are byzantine
        "NUM_KRUM" : 1, #Number of gradients to average. If NUM_KRUM = 1, basic Krum; if NUM_KRUM > 1, multi-Krum
        "TRIM_PROPORTION": 1/5,
        "SELECTION_STRATEGY": "Stratified_RS", #sample selection: Simple_RS or Stratified_RS
        "nlabels": 10,

        #Stratified kwargs
        "strata_weights":  strata_weights_unif, #stratum weights for weighted aggregation
        "conceal_pois_class": True, #Poisoned worker either conceals poisoned class
        "influence": None, #stratum to influence: "strata" + stratum number, default None
        "stratify1": 'kmeans', #stratification technique: 'median', 'kmeans', 'spectral'
        "stratify2": 'kmeans',
        "stratify_kwargs_s1": {'n_clusters' : 10, 'max_iters' : 500, 'seed' : 1234}, #Kwargs for the unsupervised strat technique
        "stratify_kwargs_s2": {'n_clusters' : 5, 'max_iters' : 500, 'seed' : 1234, 'thresh': 0.65},
        # Kwargs for the unsupervised strat technique
        "assumed_workers_per_round_stratum": int(25/2), #We divide NUM_WORKERS_PER_ROUND by the number of clusters
        "assumed_poisoned_per_round_stratum": int((10/50)*(25/2)) #Ratio of poisoned workers to num_workers applied to assumed workers per round
    }

kw_strsTMeanUnif2 = {
        "NUM_WORKERS_PER_ROUND" : 50, #4, #25
        "ASSUMED_POISONED_WORKERS_PER_ROUND" : 5, #an estimate of how many of the selected workers are byzantine
        "NUM_KRUM" : 1, #Number of gradients to average. If NUM_KRUM = 1, basic Krum; if NUM_KRUM > 1, multi-Krum
        "TRIM_PROPORTION": 1/5,
        "SELECTION_STRATEGY": "Stratified_RS", #sample selection: Simple_RS or Stratified_RS
        "nlabels": 10,

        #Stratified kwargs
        "strata_weights":  strata_weights_unif, #stratum weights for weighted aggregation
        "conceal_pois_class": True, #Poisoned worker either conceals poisoned class
        "influence": None, #stratum to influence: "strata" + stratum number, default None
        "stratify1": 'kmeans', #stratification technique: 'median', 'kmeans', 'spectral'
        "stratify2": 'spectral',
        "stratify_kwargs_s1": {'n_clusters' : 10, 'max_iters' : 500, 'seed' : 1234}, #Kwargs for the unsupervised strat technique
        "stratify_kwargs_s2": {'n_clusters': 4, 'n_neighbors' : 10, 'affinity' : 'rbf', 'random_state' : 1234, 'thresh': 0.70},
        # Kwargs for the unsupervised strat technique
        "assumed_workers_per_round_stratum": int(25/2), #We divide NUM_WORKERS_PER_ROUND by the number of clusters
        "assumed_poisoned_per_round_stratum": int((10/50)*(25/2)) #Ratio of poisoned workers to num_workers applied to assumed workers per round
    }

kw_strsTMeanDyn = {
        "NUM_WORKERS_PER_ROUND" : 50, #4, #25
        "ASSUMED_POISONED_WORKERS_PER_ROUND" : 5, #an estimate of how many of the selected workers are byzantine
        "NUM_KRUM" : 1, #Number of gradients to average. If NUM_KRUM = 1, basic Krum; if NUM_KRUM > 1, multi-Krum
        "TRIM_PROPORTION": 1/5,
        "SELECTION_STRATEGY": "Stratified_RS", #sample selection: Simple_RS or Stratified_RS
        "nlabels": 10,

        #Stratified kwargs
        "strata_weights":  strata_weights_list_dyn, #stratum weights for weighted aggregation
        "conceal_pois_class": True, #Poisoned worker either conceals poisoned class
        "influence": None, #stratum to influence: "strata" + stratum number, default None
        "stratify1": 'kmeans', #stratification technique: 'median', 'kmeans', 'spectral'
        "stratify2": 'kmeans',
        "stratify_kwargs_s1": {'n_clusters' : 10, 'max_iters' : 500, 'seed' : 1234}, #Kwargs for the unsupervised strat technique
        "stratify_kwargs_s2": {'n_clusters': 2, 'max_iters': 500, 'seed': 1234, 'thresh': 0.75},
        # Kwargs for the unsupervised strat technique
        "assumed_workers_per_round_stratum": int(25/2), #We divide NUM_WORKERS_PER_ROUND by the number of clusters
        "assumed_poisoned_per_round_stratum": int((10/50)*(25/2)) #Ratio of poisoned workers to num_workers applied to assumed workers per round
    }

kw_srsTMean = {
        "NUM_WORKERS_PER_ROUND" : 50, #4, #25
        "ASSUMED_POISONED_WORKERS_PER_ROUND" : 5, #an estimate of how many of the selected workers are byzantine
        "NUM_KRUM" : 1, #Number of gradients to average. If NUM_KRUM = 1, basic Krum; if NUM_KRUM > 1, multi-Krum
        "TRIM_PROPORTION": 1/5,
        "SELECTION_STRATEGY": "Simple_RS", #sample selection: Simple_RS or Stratified_RS
        "nlabels": 10,

        #Stratified kwargs
        "strata_weights": None, #stratum weights for weighted aggregation
        "conceal_pois_class": None, #Poisoned worker either conceals poisoned class
        "influence": None, #stratum to influence: "strata" + stratum number, default None
        "stratify1": None, #stratification technique: 'median', 'kmeans'
        "stratify2": None,
        "stratify_kwargs_s1": None, #Kwargs for the unsupervised strat technique
        "stratify_kwargs_s2": None,
        "assumed_workers_per_round_stratum": None, #We divide NUM_WORKERS_PER_ROUND by the number of clusters
        "assumed_poisoned_per_round_stratum": None #Ratio of poisoned workers to num_workers applied to assumed workers per round
    }

"""
414 - Fed Avg, poisoned
415 - Strat Trimmed Mean - Unif
416 - Strat Trimmed Mean - Dynamic
417 - SRS Trimmed Mean [Still to run]
418 - Fed Avg, clean

419 - Strat Trim Mean, Two-stage UStrat, Filtered
"""

"""
100 workers, 20 poisoned, 50 random workers to take
420 - SRS FedAvg Clean
421 - SRS FedAvg Pois
422 - SRS Trimmed Mean
423 - StRS TMean, Unif, Filtered, Metric - Fscore

424 - StRS TMean, Unif, Filtered, Metric - Precision, Recall [To implement]
"""

"""
100 workers, 20 poisoned, 50 random workers to take
520 - SRS FedAvg Clean
521 - SRS FedAvg Pois
522 - SRS Trimmed Mean
523 - StRS TMean, Unif, Filtered, Metric - Fscore

exp_ids = [520, 521, 522, 523]
num_pois = [0, 20, 20, 20]
exp_kwargs = [kw_srsFedAvg, kw_srsFedAvg, kw_srsTMean, kw_strsTMeanUnif]
agg = ["FedAvg", "FedAvg", "TrimMean", "StratTrimMean"]
"""

"""
100 workers, 20 poisoned, 50 random workers to take
620 - SRS FedAvg Clean
621 - SRS FedAvg Pois
622 - SRS Trimmed Mean
623 - StRS TMean, Unif, Filtered, Metric - Fscore

exp_ids = [620, 621, 622, 623]
num_pois = [0, 20, 20, 20]
exp_kwargs = [kw_srsFedAvg, kw_srsFedAvg, kw_srsTMean, kw_strsTMeanUnif]
agg = ["FedAvg", "FedAvg", "TrimMean", "StratTrimMean"]
"""

exp_ids = [623, 622]
num_pois = [20, 20]
exp_kwargs = [kw_strsTMeanUnif, kw_srsTMean]
agg = ["StratTrimMean", "TrimMean"]
#sources = [4]
#targets = [6]
#datasets = ['FashionMNIST']

#TO-DOs: KWARGS for the unsupervised stratification

if __name__ == '__main__':
    for num in range(len(exp_ids)):
        START_EXP_IDX = exp_ids[num] #Change here
        NUM_EXP = 1 #We can make this multiple experiments.

        NUM_POISONED_WORKERS = num_pois[num] #The total number of poisoned workers/clients
        REPLACEMENT_METHOD = replace_N_with_M
        PARAMETERS_UPLOADED = 1.0
        PARAMETERS_DOWNLOADED = 1.0 #set to 1.0 for unrestricted parameter sharing
        KWARGS = exp_kwargs[num]
        DATASET = "FashionMNIST" #datasets[num] #
        SOURCE = 6 #class to poison
        TARGET = 0 #class to flip to
        AGG = agg[num] #"StratMedian" #"FedAvg" #"StratKrum" #StratKrum" #"StratFedAvg" #MultiKrum" #"StratTrimMean" #"StratKrum" #Change here
        INIT= "Default" # "Default"
        DISTRIBUTION = 'Label Skew' #distrib[num]

        count = 0
        for experiment_id in range(START_EXP_IDX, START_EXP_IDX + NUM_EXP):
            count += 1 #experiment count
            rd.seed(1234 + 100 * count)
            np.random.seed(1234 + 100 * (count-1))
            torch.manual_seed(1234 + 100 * (count-1))
            with open("experiment_param_notes.txt", 'a') as f:
                f.write(f"{experiment_id}, {DATASET}, {NUM_POISONED_WORKERS}, {SOURCE}, {TARGET}, {get_class_label_from_num(DATASET, SOURCE)}, {get_class_label_from_num(DATASET, TARGET)}, {PARAMETERS_DOWNLOADED}, {PARAMETERS_UPLOADED}, {AGG}, {INIT}, {DISTRIBUTION}, {KWARGS['strata_weights'], KWARGS['conceal_pois_class'], KWARGS['influence'], KWARGS['stratify1'], KWARGS['stratify_kwargs_s1'], KWARGS['stratify2'], KWARGS['stratify_kwargs_s2']}\n")
            print(f"Exp ID: {experiment_id}, Num pois: {NUM_POISONED_WORKERS}, Uploaded: {PARAMETERS_UPLOADED}")
            run_exp(REPLACEMENT_METHOD, NUM_POISONED_WORKERS, KWARGS, RandomSelectionStrategy(), experiment_id, PARAMETERS_UPLOADED, PARAMETERS_DOWNLOADED, DATASET, INIT, DISTRIBUTION, SOURCE, TARGET, AGG) #Edited: I added TARGET as a parameter

    #This uses a simple randoom selection of workers using RandomSelectionStrategy()