from data_distribution_v2 import replace_N_with_M
from utilities import RandomSelectionStrategy
from saving_utils import get_class_label_from_num
import random as rd
import numpy as np
from server_core_v2 import run_exp
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
num_workers = 50
num_poisoned = 10
strata_weights_list = [
    {'strata1': 0.0, 'strata2': 1.0},
    {'strata1': 0.05, 'strata2': 0.95},
    {'strata1': 0.1, 'strata2': 0.9},
    {'strata1': 0.15, 'strata2': 0.85},
    {'strata1': 0.2, 'strata2': 0.8},
    {'strata1': 0.25, 'strata2': 0.75},
    {'strata1': 0.3, 'strata2': 0.7},
    {'strata1': 0.35, 'strata2': 0.65},
    {'strata1': 0.4, 'strata2': 0.6},
    {'strata1': 0.45, 'strata2': 0.55},
    {'strata1': 0.5, 'strata2': 0.5},
    {'strata1': 0.55, 'strata2': 0.45},
    {'strata1': 0.6, 'strata2': 0.4},
    {'strata1': 0.65, 'strata2': 0.35},
    {'strata1': 0.7, 'strata2': 0.3},
    {'strata1': 0.75, 'strata2': 0.25},
    {'strata1': 0.8, 'strata2': 0.2},
    {'strata1': 0.85, 'strata2': 0.15},
    {'strata1': 0.9, 'strata2': 0.1},
    {'strata1': 0.95, 'strata2': 0.05},
    {'strata1': 1.0, 'strata2': 0.0},
]


kw_srsFedAvg = {
        #Non-IID disbn
        "ndistr" : 2,
        "pois_distr" : {'orig': 0, 'skewed': 0},
        "NUM_WORKERS_PER_ROUND" : 25,
        "SELECTION_STRATEGY": "Simple_RS", #sample selection: Simple_RS or Stratified_RS
        "nlabels": 10,

        #Stratified kwargs
        "strata_weights": None, #stratum weights for weighted aggregation
        "conceal_pois_class": True, #Poisoned worker either conceals poisoned class
        "influence": None, #stratum to influence: "strata" + stratum number, default None
        "stratify": 'median', #stratification technique: 'median', 'kmeans'
        "nclusters": 2, #Number of clusters for the unsupervised stratification technique
        "assumed_workers_per_round_stratum": int(25/2), #We divide NUM_WORKERS_PER_ROUND by the number of clusters
        "assumed_poisoned_per_round_stratum": int((10/50)*(25/2)) #Ratio of poisoned workers to num_workers applied to assumed workers per round
    }

kw_srsFedAvgPois = {
        #Non-IID disbn (for v2)
        "ndistr" : 2,
        "pois_distr" : {'orig': num_poisoned // 2, 'skewed': num_poisoned // 2},

        "NUM_WORKERS_PER_ROUND" : 25,
        "SELECTION_STRATEGY": "Simple_RS", #sample selection: Simple_RS or Stratified_RS
        "nlabels": 10,

        #Stratified kwargs
        "strata_weights": None, #stratum weights for weighted aggregation
        "conceal_pois_class": None, #Poisoned worker either conceals poisoned class
        "influence": None, #stratum to influence: "strata" + stratum number, default None
        "stratify": None, #stratification technique: 'median', 'kmeans'
        "nclusters": None, #Number of clusters for the unsupervised stratification technique
    }

kw_strsMed = {
        #Non-IID disbn
        "ndistr" : 2, #number of actual distribution
        "pois_distr" : {'orig': num_poisoned // 2, 'skewed': num_poisoned // 2}, #number of poisoned workers

        "NUM_WORKERS_PER_ROUND" : 25, #4, #25
        "ASSUMED_POISONED_WORKERS_PER_ROUND" : 5, #an estimate of how many of the selected workers are byzantine
        "NUM_KRUM" : 1, #Number of gradients to average. If NUM_KRUM = 1, basic Krum; if NUM_KRUM > 1, multi-Krum
        "TRIM_PROPORTION": 1/5,
        "SELECTION_STRATEGY": "Stratified_RS", #sample selection: Simple_RS or Stratified_RS
        "nlabels": 10,

        #Stratified kwargs
        "strata_weights": strata_weights_list, #stratum weights for weighted aggregation
        "conceal_pois_class": True, #Poisoned worker either conceals poisoned class
        "influence": None, #stratum to influence: "strata" + stratum number, default None
        "stratify": 'median', #stratification technique: 'median', 'kmeans'
        "nclusters": 2, #Number of clusters for the unsupervised stratification technique
        "assumed_workers_per_round_stratum": int(25/2), #We divide NUM_WORKERS_PER_ROUND by the number of clusters
        "assumed_poisoned_per_round_stratum": int((10/50)*(25/2)) #Ratio of poisoned workers to num_workers applied to assumed workers per round
    }

kw_strsTMean = KWARGS = {
        #Non-IID disbn
        "ndistr" : 2,
        "pois_distr" : {'orig': num_poisoned // 2, 'skewed': num_poisoned // 2},

        "NUM_WORKERS_PER_ROUND" : 25, #4, #25
        "ASSUMED_POISONED_WORKERS_PER_ROUND" : 5, #an estimate of how many of the selected workers are byzantine
        "NUM_KRUM" : 1, #Number of gradients to average. If NUM_KRUM = 1, basic Krum; if NUM_KRUM > 1, multi-Krum
        "TRIM_PROPORTION": 1/5,
        "SELECTION_STRATEGY": "Stratified_RS", #sample selection: Simple_RS or Stratified_RS
        "nlabels": 10,

        #Stratified kwargs
        "strata_weights":  strata_weights_list, #stratum weights for weighted aggregation
        "conceal_pois_class": True, #Poisoned worker either conceals poisoned class
        "influence": None, #stratum to influence: "strata" + stratum number, default None
        "stratify": 'median', #stratification technique: 'median', 'kmeans'
        "nclusters": 2, #Number of clusters for the unsupervised stratification technique
        "assumed_workers_per_round_stratum": int(25/2), #We divide NUM_WORKERS_PER_ROUND by the number of clusters
        "assumed_poisoned_per_round_stratum": int((10/50)*(25/2)) #Ratio of poisoned workers to num_workers applied to assumed workers per round
    }

kw_srsTMean = {
        #Non-IID disbn
        "ndistr" : 2,
        "pois_distr" : {'orig': num_poisoned // 2, 'skewed': num_poisoned // 2},

        "NUM_WORKERS_PER_ROUND" : 25, #4, #25
        "ASSUMED_POISONED_WORKERS_PER_ROUND" : 5, #an estimate of how many of the selected workers are byzantine
        "NUM_KRUM" : 1, #Number of gradients to average. If NUM_KRUM = 1, basic Krum; if NUM_KRUM > 1, multi-Krum
        "TRIM_PROPORTION": 1/5,
        "SELECTION_STRATEGY": "Simple_RS", #sample selection: Simple_RS or Stratified_RS
        "nlabels": 10,

        #Stratified kwargs
        "strata_weights": None, #stratum weights for weighted aggregation
        "conceal_pois_class": True, #Poisoned worker either conceals poisoned class
        "influence": None, #stratum to influence: "strata" + stratum number, default None
        "stratify": 'median', #stratification technique: 'median', 'kmeans'
        "nclusters": 2, #Number of clusters for the unsupervised stratification technique
        "assumed_workers_per_round_stratum": int(25/2), #We divide NUM_WORKERS_PER_ROUND by the number of clusters
        "assumed_poisoned_per_round_stratum": int((10/50)*(25/2)) #Ratio of poisoned workers to num_workers applied to assumed workers per round
    }

exp_ids = [412]
#num_pois = [10, 10, 10, 10, 10]
#exp_kwargs = [kw_srsFedAvgPois, kw_srsFedAvgPois, kw_srsFedAvgPois, kw_srsFedAvgPois, kw_srsFedAvgPois]
#agg = ["FedAvg", "FedAvg", "FedAvg", "FedAvg", "FedAvg"]
#distrib = ['Non-IID_v2', 'Non-IID_v2', 'Non-IID_v2', 'Non-IID_v2', 'Non-IID_v2']
sources = [4]
targets = [6]
datasets = ['FashionMNIST']

if __name__ == '__main__':
    for num in range(len(exp_ids)):
        START_EXP_IDX = exp_ids[num] #Change here
        NUM_EXP = 1 #We can make this multiple experiments.

        NUM_POISONED_WORKERS = 10 #num_pois[num] #The total number of poisoned workers/clients
        REPLACEMENT_METHOD = replace_N_with_M
        PARAMETERS_UPLOADED = 1.0
        PARAMETERS_DOWNLOADED = 1.0 #set to 1.0 for unrestricted parameter sharing
        KWARGS = kw_srsFedAvgPois #exp_kwargs[num]
        DATASET = datasets[num] #"FashionMNIST" #
        TARGET = targets[num] #The target label as the poisoning/flipping
        SOURCE = sources[num] #poisoned class
        AGG = "FedAvg" #agg[num] #"StratMedian" #"FedAvg" #"StratKrum" #StratKrum" #"StratFedAvg" #MultiKrum" #"StratTrimMean" #"StratKrum" #Change here
        INIT= "Default" # "Default"
        DISTRIBUTION = 'Label Skew' #distrib[num]

        count = 0
        for experiment_id in range(START_EXP_IDX, START_EXP_IDX + NUM_EXP):
            count += 1 #experiment count
            rd.seed(1234 + 100 * count)
            np.random.seed(1234 + 100 * (count-1))
            torch.manual_seed(1234 + 100 * (count-1))
            with open("experiment_param_notes.txt", 'a') as f:
                f.write(f"{experiment_id}, {DATASET}, {NUM_POISONED_WORKERS}, {SOURCE}, {TARGET}, {get_class_label_from_num(DATASET, SOURCE)}, {get_class_label_from_num(DATASET, TARGET)}, {PARAMETERS_DOWNLOADED}, {PARAMETERS_UPLOADED}, {AGG}, {INIT}, {DISTRIBUTION}, {KWARGS['strata_weights'], KWARGS['conceal_pois_class'], KWARGS['influence'], KWARGS['stratify'], KWARGS['nclusters']}\n")
            print(f"Exp ID: {experiment_id}, Num pois: {NUM_POISONED_WORKERS}, Uploaded: {PARAMETERS_UPLOADED}")
            run_exp(REPLACEMENT_METHOD, NUM_POISONED_WORKERS, KWARGS, RandomSelectionStrategy(), experiment_id, PARAMETERS_UPLOADED, PARAMETERS_DOWNLOADED, DATASET, INIT, DISTRIBUTION, SOURCE, TARGET, AGG) #Edited: I added TARGET as a parameter

    #This uses a simple randoom selection of workers using RandomSelectionStrategy()