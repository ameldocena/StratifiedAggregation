from data_distribution import replace_N_with_M
from utilities import RandomSelectionStrategy
from saving_utils import get_class_label_from_num
import random as rd
import numpy as np
from server_core import run_exp

#START_EXP_IDX
exper = dict()
#exper['index'] = [956, 957, 958, 959, 960, 961] #MKrum-Horded-Unif, Median-Horded-Unif, TMean-Horded-Unif
#exper['pois'] = [0, 10, 10]

# This block can be set as constant
#exper['data'] = ['CIFAR10']

#UPNEXT: UNSUPERVISED STRATIFICATION

#Keyword arguments
kw_srsFedAvg = {"NUM_WORKERS_PER_ROUND" : 25,
    "SELECTION_STRATEGY": "Simple_RS",
    "strata_weights": None}

kw_strsFedAvg = {"NUM_WORKERS_PER_ROUND" : 25, #4, #25
    "NUM_KRUM": 1,  # Number of gradients to average. If NUM_KRUM = 1, basic Krum; if NUM_KRUM > 1, multi-Krum
    "SELECTION_STRATEGY": "Stratified_RS",  # sample selection: Simple_RS or Stratified_RS
    "nlabels": 10,
    #Stratified kwargs

    "strata_weights": {'strata1': 0.25, 'strata2': 0.25, 'strata3': 0.25, 'strata4': 0.25},
    # stratum weights for weighted aggregation
    "conceal_pois_class": True,  # Poisoned worker either conceals poisoned class
    "influence": None,  # stratum to influence: "strata" + stratum number, default None
    "stratify": 'median',  # stratification technique: 'median', 'kmeans'
    "nclusters": 4,  # Number of clusters for the unsupervised stratification technique
    "assumed_workers_per_round_stratum": int(25 / 4),  # We divide NUM_WORKERS_PER_ROUND by the number of clusters
    "assumed_poisoned_per_round_stratum": int((10 / 50) * (25 / 4))
    }

kw_strsMKrumHorded = {"NUM_WORKERS_PER_ROUND" : 25, #4, #25
    "NUM_KRUM": 3,  # Number of gradients to average. If NUM_KRUM = 1, basic Krum; if NUM_KRUM > 1, multi-Krum
    "SELECTION_STRATEGY": "Stratified_RS",  # sample selection: Simple_RS or Stratified_RS
    "nlabels": 10,
    #Stratified kwargs

    "strata_weights": {'strata1': 0.25, 'strata2': 0.25, 'strata3': 0.25, 'strata4': 0.25},
    # stratum weights for weighted aggregation
    "conceal_pois_class": True,  # Poisoned worker either conceals poisoned class
    "influence": 'strata1',  # stratum to influence: "strata" + stratum number, default None
    "stratify": 'median',  # stratification technique: 'median', 'kmeans'
    "nclusters": 4,  # Number of clusters for the unsupervised stratification technique
    "assumed_workers_per_round_stratum": int(25 / 4),  # We divide NUM_WORKERS_PER_ROUND by the number of clusters
    "assumed_poisoned_per_round_stratum": int((10 / 50) * (25 / 4))
    }

kw_strsMedianHorded = {"NUM_WORKERS_PER_ROUND" : 25, #4, #25
    "SELECTION_STRATEGY": "Stratified_RS",  # sample selection: Simple_RS or Stratified_RS
    "nlabels": 10,
    "NUM_KRUM": 1,
    #Stratified kwargs
    "strata_weights": {'strata1': 0.25, 'strata2': 0.25, 'strata3': 0.25, 'strata4': 0.25},
    # stratum weights for weighted aggregation
    "conceal_pois_class": True,  # Poisoned worker either conceals poisoned class
    "influence": 'strata1',  # stratum to influence: "strata" + stratum number, default None
    "stratify": 'median',  # stratification technique: 'median', 'kmeans'
    "nclusters": 4,  # Number of clusters for the unsupervised stratification technique
    "assumed_workers_per_round_stratum": int(25 / 4),
    # We divide NUM_WORKERS_PER_ROUND by the number of clusters
    "assumed_poisoned_per_round_stratum": int((10 / 50) * (25 / 4))
   }

kw_strsTMeanHorded = {"NUM_WORKERS_PER_ROUND" : 25, #4, #25
    "SELECTION_STRATEGY": "Stratified_RS",  # sample selection: Simple_RS or Stratified_RS
    "nlabels": 10,
    "TRIM_PROPORTION": 1/5,
    "NUM_KRUM": 1,
    #Stratified kwargs
    "strata_weights": {'strata1': 0.25, 'strata2': 0.25, 'strata3': 0.25, 'strata4': 0.25},
    # stratum weights for weighted aggregation
    "conceal_pois_class": True,  # Poisoned worker either conceals poisoned class
    "influence": 'strata1',  # stratum to influence: "strata" + stratum number, default None
    "stratify": 'median',  # stratification technique: 'median', 'kmeans'
    "nclusters": 4,  # Number of clusters for the unsupervised stratification technique
    "assumed_workers_per_round_stratum": int(25 / 4),
    # We divide NUM_WORKERS_PER_ROUND by the number of clusters
    "assumed_poisoned_per_round_stratum": int((10 / 50) * (25 / 4))
    }

# Poisoned data - Krum
kw_srsKrum = {"NUM_WORKERS_PER_ROUND": 25,
              "ASSUMED_POISONED_WORKERS_PER_ROUND" : 5,
              "NUM_KRUM": 1,  # Number of gradients to average. If NUM_KRUM = 1, basic Krum; if NUM_KRUM > 1, multi-Krum
              "SELECTION_STRATEGY": "Simple_RS",  # sample selection: Simple_RS or Stratified_RS
              "nlabels": 10,

              # Stratified kwargs
              "strata_weights": {'strata1': 0.25, 'strata2': 0.25, 'strata3': 0.25, 'strata4': 0.25},
              # stratum weights for weighted aggregation
              "conceal_pois_class": True,  # Poisoned worker either conceals poisoned class
              "influence": None,  # 'strata1',  # stratum to influence: "strata" + stratum number, default None
              "stratify": 'median',  # stratification technique: 'median', 'kmeans'
              "nclusters": 4,  # Number of clusters for the unsupervised stratification technique
              "assumed_workers_per_round_stratum": int(25 / 4),
              # We divide NUM_WORKERS_PER_ROUND by the number of clusters
              "assumed_poisoned_per_round_stratum": int((10 / 50) * (25 / 4))
              }

kw_strsKrum = {"NUM_WORKERS_PER_ROUND": 25,
               "ASSUMED_POISONED_WORKERS_PER_ROUND" : 5,
               "NUM_KRUM": 1,
               # Number of gradients to average. If NUM_KRUM = 1, basic Krum; if NUM_KRUM > 1, multi-Krum
               "SELECTION_STRATEGY": "Stratified_RS",  # sample selection: Simple_RS or Stratified_RS
               "nlabels": 10,

               # Stratified kwargs
               "strata_weights": {'strata1': 0.25, 'strata2': 0.25, 'strata3': 0.25, 'strata4': 0.25},
               # stratum weights for weighted aggregation
               "conceal_pois_class": True,  # Poisoned worker either conceals poisoned class
               "influence": None,  # 'strata1',  # stratum to influence: "strata" + stratum number, default None
               "stratify": 'median',  # stratification technique: 'median', 'kmeans'
               "nclusters": 4,  # Number of clusters for the unsupervised stratification technique
               "assumed_workers_per_round_stratum": int(25 / 4),
               # We divide NUM_WORKERS_PER_ROUND by the number of clusters
               "assumed_poisoned_per_round_stratum": int((10 / 50) * (25 / 4))
               }

# Poisoned data - MKrum
kw_srsMKrum = {"NUM_WORKERS_PER_ROUND": 25,
               "ASSUMED_POISONED_WORKERS_PER_ROUND" : 5,
               "NUM_KRUM": 3,
               # Number of gradients to average. If NUM_KRUM = 1, basic Krum; if NUM_KRUM > 1, multi-Krum
               "SELECTION_STRATEGY": "Simple_RS",  # sample selection: Simple_RS or Stratified_RS
               "nlabels": 10,

               # Stratified kwargs
               "strata_weights": {'strata1': 0.25, 'strata2': 0.25, 'strata3': 0.25, 'strata4': 0.25},
               # stratum weights for weighted aggregation
               "conceal_pois_class": True,  # Poisoned worker either conceals poisoned class
               "influence": None, # 'strata1',  # stratum to influence: "strata" + stratum number, default None
               "stratify": 'median',  # stratification technique: 'median', 'kmeans'
               "nclusters": 4,  # Number of clusters for the unsupervised stratification technique
               "assumed_workers_per_round_stratum": int(25 / 4),
               # We divide NUM_WORKERS_PER_ROUND by the number of clusters
               "assumed_poisoned_per_round_stratum": int((10 / 50) * (25 / 4))
               }

kw_strsMKrum = {"NUM_WORKERS_PER_ROUND": 25,
                "ASSUMED_POISONED_WORKERS_PER_ROUND" : 5,
                "NUM_KRUM": 3,
                # Number of gradients to average. If NUM_KRUM = 1, basic Krum; if NUM_KRUM > 1, multi-Krum
                "SELECTION_STRATEGY": "Stratified_RS",  # sample selection: Simple_RS or Stratified_RS
                "nlabels": 10,

                # Stratified kwargs
                "strata_weights": {'strata1': 0.25, 'strata2': 0.25, 'strata3': 0.25, 'strata4': 0.25},
                # stratum weights for weighted aggregation
                "conceal_pois_class": True,  # Poisoned worker either conceals poisoned class
                "influence": None,  # 'strata1',  # stratum to influence: "strata" + stratum number, default None
                "stratify": 'median',  # stratification technique: 'median', 'kmeans'
                "nclusters": 4,  # Number of clusters for the unsupervised stratification technique
                "assumed_workers_per_round_stratum": int(25 / 4),
                # We divide NUM_WORKERS_PER_ROUND by the number of clusters
                "assumed_poisoned_per_round_stratum": int((10 / 50) * (25 / 4))
                }

# Poisoned data - Median
kw_srsMed = {"NUM_WORKERS_PER_ROUND": 25,
             "ASSUMED_POISONED_WORKERS_PER_ROUND" : 5,
             "NUM_KRUM": 1,  # Number of gradients to average. If NUM_KRUM = 1, basic Krum; if NUM_KRUM > 1, multi-Krum
             "SELECTION_STRATEGY": "Simple_RS",  # sample selection: Simple_RS or Stratified_RS
             "nlabels": 10,

             # Stratified kwargs
             "strata_weights": {'strata1': 0.25, 'strata2': 0.25, 'strata3': 0.25, 'strata4': 0.25},
             # stratum weights for weighted aggregation
             "conceal_pois_class": True,  # Poisoned worker either conceals poisoned class
             "influence": None,  # 'strata1',  # stratum to influence: "strata" + stratum number, default None
             "stratify": 'median',  # stratification technique: 'median', 'kmeans'
             "nclusters": 4,  # Number of clusters for the unsupervised stratification technique
             "assumed_workers_per_round_stratum": int(25 / 4),
             # We divide NUM_WORKERS_PER_ROUND by the number of clusters
             "assumed_poisoned_per_round_stratum": int((10 / 50) * (25 / 4))
             }

kw_strsMed = {"NUM_WORKERS_PER_ROUND": 25,
              "ASSUMED_POISONED_WORKERS_PER_ROUND" : 5,
              "NUM_KRUM": 1,  # Number of gradients to average. If NUM_KRUM = 1, basic Krum; if NUM_KRUM > 1, multi-Krum
              "SELECTION_STRATEGY": "Stratified_RS",  # sample selection: Simple_RS or Stratified_RS
              "nlabels": 10,

              # Stratified kwargs
              "strata_weights": {'strata1': 0.25, 'strata2': 0.25, 'strata3': 0.25, 'strata4': 0.25},
              # stratum weights for weighted aggregation
              "conceal_pois_class": True,  # Poisoned worker either conceals poisoned class
              "influence": None,  # 'strata1',  # stratum to influence: "strata" + stratum number, default None
              "stratify": 'median',  # stratification technique: 'median', 'kmeans'
              "nclusters": 4,  # Number of clusters for the unsupervised stratification technique
              "assumed_workers_per_round_stratum": int(25 / 4),
              # We divide NUM_WORKERS_PER_ROUND by the number of clusters
              "assumed_poisoned_per_round_stratum": int((10 / 50) * (25 / 4))
              }

# Poisoned data - Trimmed Mean
kw_srsTMean = {"NUM_WORKERS_PER_ROUND": 25,
               "ASSUMED_POISONED_WORKERS_PER_ROUND" : 5,
               "NUM_KRUM": 1,
               # Number of gradients to average. If NUM_KRUM = 1, basic Krum; if NUM_KRUM > 1, multi-Krum
               "SELECTION_STRATEGY": "Simple_RS",  # sample selection: Simple_RS or Stratified_RS
               "nlabels": 10,
               "TRIM_PROPORTION": 1 / 5,

               # Stratified kwargs
               "strata_weights": {'strata1': 0.25, 'strata2': 0.25, 'strata3': 0.25, 'strata4': 0.25},
               # stratum weights for weighted aggregation
               "conceal_pois_class": True,  # Poisoned worker either conceals poisoned class
               "influence": None,  # 'strata1',  # stratum to influence: "strata" + stratum number, default None
               "stratify": 'median',  # stratification technique: 'median', 'kmeans'
               "nclusters": 4,  # Number of clusters for the unsupervised stratification technique
               "assumed_workers_per_round_stratum": int(25 / 4),
               # We divide NUM_WORKERS_PER_ROUND by the number of clusters
               "assumed_poisoned_per_round_stratum": int((10 / 50) * (25 / 4))
               }

kw_strsTMean = {"NUM_WORKERS_PER_ROUND": 25,
                "ASSUMED_POISONED_WORKERS_PER_ROUND" : 5,
                "NUM_KRUM": 1,
                # Number of gradients to average. If NUM_KRUM = 1, basic Krum; if NUM_KRUM > 1, multi-Krum
                "SELECTION_STRATEGY": "Stratified_RS",  # sample selection: Simple_RS or Stratified_RS
                "nlabels": 10,
                "TRIM_PROPORTION": 1 / 5,

                # Stratified kwargs
                "strata_weights": {'strata1': 0.25, 'strata2': 0.25, 'strata3': 0.25, 'strata4': 0.25},
                # stratum weights for weighted aggregation
                "conceal_pois_class": True,  # Poisoned worker either conceals poisoned class
                "influence": None,  # 'strata1',  # stratum to influence: "strata" + stratum number, default None
                "stratify": 'kmeans',  # stratification technique: 'median', 'kmeans'
                "nclusters": 3,  # Number of clusters for the unsupervised stratification technique
                "assumed_workers_per_round_stratum": int(25 / 4),
                # We divide NUM_WORKERS_PER_ROUND by the number of clusters
                "assumed_poisoned_per_round_stratum": int((10 / 50) * (25 / 4))
                }

#CIFAR-10, Stratified FedAvg, Clean and Poisoned, 3 label-flipped trials
exper['index'] = [953, 954, 955, 959, 960, 961]
exper['pois'] = [0, 0, 0, 10, 10, 10]

exper['source'] = [0, 1, 5,
                   0, 1, 5]

exper['target'] = [2, 9, 3,
                   2, 9, 3]

exper['agg'] = ['StratFedAvg', 'StratFedAvg', 'StratFedAvg',
                'StratFedAvg', 'StratFedAvg', 'StratFedAvg']

exper['kwargs'] = [kw_strsFedAvg, kw_strsFedAvg, kw_strsFedAvg,
                   kw_strsFedAvg, kw_strsFedAvg, kw_strsFedAvg]

if __name__ == '__main__':
    for num in range(len(exper['index'])):
        START_EXP_IDX = exper['index'][num]  # Change here
        NUM_EXP = 1  # We can make this multiple experiments.
        NUM_POISONED_WORKERS = exper['pois'][num]  # The total number of poisoned workers/clients
        REPLACEMENT_METHOD = replace_N_with_M
        PARAMETERS_UPLOADED = 1.0
        PARAMETERS_DOWNLOADED = 1.0  # set to 1.0 for unrestricted parameter sharing
        KWARGS = exper['kwargs'][num]
        DATASET = "CIFAR10"  #
        TARGET = exper['target'][num]  # The target label as the poisoning/flipping
        SOURCE = exper['source'][num]  # poisoned class
        AGG = exper['agg'][num]  # StratKrum" #"StratFedAvg" #MultiKrum" #"StratTrimMean" #"StratKrum" #Change here
        INIT = "Randomized"  # "Default"
        DISTRIBUTION = "Non-IID_v2"

        count = 0
        for experiment_id in range(START_EXP_IDX, START_EXP_IDX + NUM_EXP):
            count += 1  #experiment count
            rd.seed(1234 + 100 * count)
            np.random.seed(1234 + 100 * (count - 1))
            with open("experiment_param_notes.txt", 'a') as f:
                f.write(f"{experiment_id}, {DATASET}, {NUM_POISONED_WORKERS}, {SOURCE}, {TARGET}, {get_class_label_from_num(DATASET, SOURCE)}, {get_class_label_from_num(DATASET, TARGET)}, {PARAMETERS_DOWNLOADED}, {PARAMETERS_UPLOADED}, {AGG}, {INIT}, {DISTRIBUTION}\n")
            print(f"Exp ID: {experiment_id}, Num pois: {NUM_POISONED_WORKERS}, Uploaded: {PARAMETERS_UPLOADED}")
            run_exp(REPLACEMENT_METHOD, NUM_POISONED_WORKERS, KWARGS, RandomSelectionStrategy(), experiment_id,
                    PARAMETERS_UPLOADED, PARAMETERS_DOWNLOADED, DATASET, INIT, DISTRIBUTION, SOURCE, TARGET, AGG)
