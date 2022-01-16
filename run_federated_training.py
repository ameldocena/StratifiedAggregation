from data_distribution import replace_N_with_M
from utilities import RandomSelectionStrategy
from saving_utils import get_class_label_from_num
import random as rd
import numpy as np
from server_core import run_exp



"""
Aug 23:
510 - Non-IID_v2, FMNIST
520 - IID
530 - Non-IID

600 - Non-IID_v2, FMNIST, Simple_RS
610 - Stratified_RS: Workers = 10, worker

620 - Simple RS: 50 Workers, 16 sampled per round
630 - Strat RS: same settings

700 - Simple RS (Redone)
710 - Strat RS: 4 strata
720 - Strat RS: k1
730 - Strat RS: k2
740 - Strat RS: k3
750 - Strat RS: k4
760 - Strat RS: k1 + k3
770 - Strat RS: k2 + k4
780 - Strat RS: k1 + k2
790 - Strat RS: k3 + k4
795 - Strat RS: k1 + k2 + k3
796 - Strat RS: Strat SRS Unif
797 - Strat RS: Strat SRS [w1: 0.45, w2: 0.05, w3: 0.45, w4: 0.05] *top
798 - Strat RS: Strat SRS [w1: 0.40, w2: 0.10, w3: 0.40, w4: 0.10] *top

810 - Strat RS (S1- attr skew): 4 strata
820 - Strat RS: k1
830 - Strat RS: k2
840 - Strat RS: k3
850 - Strat RS: k4
860 - Strat RS: k1 + k3
870 - Strat RS: k2 + k4
880 - Strat RS: k1 + k2
890 - Strat RS: k3 + k4
895 - Strat RS: k1 + k2 + k3
899 - Strat RS: k1 + k2 + k4

Stratification 3: S1 Attribute, S2 Quantity, S3 Label
910 - Strat RS: All 8 strata
915 - Strat RS: k4
920 - Strat RS: k1 + k2 + k3 + k4

CIFAR10
10 - Simple RS
11 - Strat RS
12 - Strat RS: k1 + k3 

Clean FMNIST
30 - Strat RS - FedAvg - Unif
31 - Strat RS - FedAvg - [w1: 0.40, w2: 0.10, w3: 0.40, w4: 0.10]
32 - Strat RS - FedAvg - [w1: 0.10, w2: 0.40, w3: 0.40, w4: 0.10]
33 - Strat RS - FedAvg - [w1: 0.30, w2: 0.20, w3: 0.30, w4: 0.20]

*Poisoned FMNIST
35 - Strat RS - FedAvg - Unif


Poisoned FMNIST
1 - Simple RS (Redone)
2 - Strat RS
3 - Strat RS: k1
4 - Strat RS: k2
5 - Strat RS: k3
6 - Strat RS: k4
7 - Strat RS: k1 + k3
8 - Strat RS: k2 + k4
9 - Strat RS: k1 + k2
20 - Strat RS: k3 + k4
21 - Strat RS: k1 + k2 + k3
22 - Strat RS: K1 + k2 + k4

23 - Strat RS (S1 - attr skew, S2 - label skew): All
24 - Strat RS (S1 - attr skew/label skew, S2 - quant skew): All

*Krum 200 epochs
40 - Simple RS 
41 - Strat RS (S1 - Attr skew, S2 - Quant skew)
42 - Strat RS (S1 - Attr skew, S2 - Label skew)
43 - Strat RS (S1 - Attr skew/label skew, S2 - Quant skew)

*MultiKrum - 5, 200 epochs
415 - Strat RS (S1 - Attr skew, S2 - Quant skew)
435 - Strat RS (S1 - Attr skew/label skew, S2 - Quant skew)

*StratifiedKrum
51 - Strat RS - Krum 1: uniform strata weights (Redone)
52 - Strat RS - Krum 3: uniform strata weights (Redone)
53 - Strat RS - Krum 3: [w1: 0.45, w2: 0.05, w3: 0.45, w4: 0.05] (Redone)
54 - Strat RS - Krum 3: [w1: 0.40, w2: 0.10, w3: 0.40, w4: 0.10] (Redone)
55 - Strat RS - Krum 3: [w1: 0.30, w2: 0.20, w3: 0.30, w4: 0.20] (Redone)
56 - [w1:35 w2:0.15 w3:0.35 w4: 0.15]
57 - [w1: 0.70, w2:0.05, w3: 0.20, w4: 0.05]
58 - [w1: 0.60, w2:0.10, w3: 0.25, w4: 0.05]

**(FALSE) CONCEALMENT OF POISONED CLASS
511 - Strat RS - Krum 1: uniform weights
512 - Strat RS - Krum 3: uniform
513 - Strat RS - Krum 3: [w1: 0.45, w2: 0.05, w3: 0.45, w4: 0.05]
514 - Strat RS - Krum 3: [w1: 0.40, w2: 0.10, w3: 0.40, w4: 0.10]
515 - Strat RS - Krum 3: [w1: 0.30, w2: 0.20, w3: 0.30, w4: 0.20]
516 - Strat RS - Krum 3: [w1: 0.20, w2: 0.30, w3: 0.30, w4: 0.20]
517 - Strat RS - Krum 3: [w1: 0.10, w2: 0.40, w3: 0.40, w4: 0.10]
518 - Strat RS - Krum 3: [w1: 0.05, w2: 0.45, w3: 0.45, w4: 0.05]

(In addition poisoned workers are able to influence to specific stratum: strata1)
518 - Strat RS - Krum 1: uniform
519 - Strat RS - Krum 3: uniform
520 - Strat RS - Krum 3: [w1: 0.05, w2: 0.30, w3: 0.35, w4: 0.30]
521 - Strat RS - Krum 3: [w1: 0.0, w2: 0.30, w3: 0.40, w4: 0.30]
522 - Strat RS - Krum 3: [w1: 0.0, w2: 0.25, w3: 0.50, w4: 0.25]
523 - Strat RS - Krum 3: [w1: 0.0, w2: 0.20, w3: 0.60, w4: 0.20]

*Median
60 - SRS - Median
61 - Strat RS - Median
62 - Strat RS - StratMedian - Uniform
63 - Strat RS - StratMedian - [w1: 0.45, w2: 0.05, w3: 0.45, w4: 0.05]
64 - Strat RS - StratMedian - [w1: 0.40, w2: 0.10, w3: 0.40, w4: 0.10]

*Trimmed Mean
70 - SRS - Trimmed Mean (1/5)
71 - Strat RS - Trimmed Mean (1/5)
72 - Strat RS - Strat Trimmed Mean (1/5) - Uniform
73 - Strat RS - Strat Trimmed Mean - [w1: 0.45, w2: 0.05, w3: 0.45, w4: 0.05]
74 - Strat RS - Strat Trimmed Mean - [w1: 0.40, w2: 0.10, w3: 0.40, w4: 0.10]
75 - 
*KMeans - Krum - Concealed
81 - Strat RS - Krum 1 - Uniform
82 - Strat RS - MKrum 3 - Uniform
83 - Strat RS - MKrum 3 - [w1: 0.20, w2: 0.40, w3: 0.40]
84 - Strat RS - MKrum 3 - [w1: 0.40, w2: 0.20, w3: 0.40]
85 - Strat RS - MKrum 3 - [w1: 0.40, w2: 0.40, w3: 0.20]
86 - Strat RS - MKRum3 - [w1: 0.20, w2: 0.50, w3: 0.30]


*CIFAR10
100 - SRS - FedAvg - Clean
101 - SRS - FedAvg - Poisoned
102 - SRS - Krum
103 - SRS - MKrum3

200 - StRS - FedAvg - Clean
201 - StRS - FedAvg - Poisoned
202 - StRS - StKrum - Unif
203 - StRS - StMKrum3 - Unif
204 - SRS - Krum
205 - StRS - Krum - Unif - Adj workers and pois_workers per round

*FMNIST - Poisoned workers influence to be in first stratum under median stratification
524 - [1, 0, 0, 0] - 69%
525 - [0, 1, 0, 0] - 46%
526 - [0, 0, 1, 0] - 47%
527 - [0, 0, 0, 1] - 65%

528 - [0.40, 0.10, 0.10, 0.40] - 69%
529 - [0.45, 0.05, 0.05, 0.45] - 69%
530 - [0.35, 0.15, 0.15, 0.35] - 68%
531 - [0.45, 0.10, 0.10, 0.35] - 69%
532 - [0.50, 0.10, 0.10, 0.30] - 69% 6933/10000
533 - [0.55, 0.10, 0.10, 0.25] - 69% 6945/10000
534 - [0.55, 0.05, 0.05, 0.30] - 69% 6935/10000
535 - [0.60, 0.10, 0.10, 0.20] - 70% 6956/10000
536 - [0.65, 0.10, 0.10, 0.15] - 70% 6984/10000

#By pairs
537 - [0.50, 0.0, 0.0, 0.50] 6903/10000 (69%)
538 - [0.50, 0.50, 0.0, 0.0] 6844/10000 (68%)
539 - [0.50, 0.0, 0.50, 0.0] 6872/10000 (69%)

1 - 4 - 3 are best combo

540 - [0.0, 0.50, 0.0, 0.50] 6518/10000 (65%)
541 - [0.0, 0.0, 0.50, 0.50] 6593/10000 (66%)

4 - 3 is the better combo
Thus, the proportion would be 1-4-3-2.
551 - [0.50, 0.10, 0.10, 0.30] 6933/10000 (69%)
552 - [0.55, 0.10, 0.05, 0.30] 6942/10000 (69%)
553 - [0.50, 0.10, 0.05, 0.35] 6913/10000 (69%)
554 - [0.60, 0.10, 0.10, 0.20] 6956/10000 (70%)
555 - [0.65, 0.10, 0.05, 0.20] 6959/10000 (70%)
556 - [0.60, 0.10, 0.05, 0.25] 6944/10000 (69%)
557 - [0.70, 0.05, 0.05, 0.20] 6966/10000 (70%)
558 - [0.73, 0.05, 0.06, 0.16] 6974/10000 (70%) **

#When penalty/weights exceed 1.0
#sum of weights/penalty exceeds 1.0
545 - [0.60, 0.10, 0.10, 0.30] 6977/10000 (70%)
546 - [0.65, 0.05, 0.10, 0.30] 6996/10000 (70%)
547 - [0.60, 0.05, 0.10, 0.35] 6977/10000 (70%)

548 - [0.65, 0.10, 0.05, 0.30] 6974/10000 (70%)
549 - [0.60, 0.10, 0.05, 0.35] 6971/10000 (70%)
550 - [0.63, 0.02, 0.02, 0.33] 6948/10000 (69%)

542 - [0.60, 0.30, 0.20, 0.10] 6981/10000 (70%)
543 - [0.65, 0.35, 0.15, 0.05] 6963/10000 (70%)
544 - [0.65, 0.30, 0.10, 0.10] 6965/10000 (70%)
545 - [0.60, 0.10, 0.20, 0.30] 7001/10000 (70%)
548 - [0.70, 0.10, 0.20, 0.50]

#Restricted

#Check the best hyperparameters for the flip and jitter, FMNIST
600 - FedAvg - 100% horizontal flip 6411/10000 (64%)
601 - FedAvg - 80% Hflip, 20% jitter 6399/10000 (64%)
602 - FedAvg - 60 Hflip, 40 jitter 6402/10000 (64%)
603 - FedAvg - 50 Hflip, 50 jitter 6390/10000 (64%)
604 - FedAvg - 40 Hflip, 60 jitter 6377/10000 (64%)
605 - FedAvg - 20 Hflip, 80 jitter 6378/10000 (64%)
606 - FedAvg - 0 Hflip, 100 jitter 6356/10000 (64%)
607 - FedAvg - 0 Hflip, 0 jitter 6411/10000 (64%)
608 - FedAvg - 1.0 Vflip, 0 jitter 6433/10000 (64%)

609 - SRS FedAvg - 1.0 Hflip, poisoned 5282/10000 (53%)
610 - SRS FedAvg - 1.0 Vflip, poisoned 5248/10000 (52%)

611 - StRS FedAvg - 1.0 Hflip, clean 6253/10000 (63%)

#Start from here
*0.50 - Horiz and Vert Flips
612 - SRS FedAvg - 0.5 Vflip, 0.5 Hflip 6428/10000 (64%). 7555/10000 (76%)
613 - SRS FedAVg - 612 + poisoned 5264/10000 (53%)
614 - SRS Krum - 613 + Krum 5725/10000 (57%)
615 - SRS MKrum3 - 6544/10000 (65%)
616 - SRS Median - 6695/10000 (67%)
617 - SRS Trimmed Mean - 7227/10000 (72%)

622 - StRS FedAvg - Clean - 7496/10000 (75%)
618 - StRS Krum - Unif - 7230/10000 (72%)
619 - StRS MKrum3 - Unif - 7808/10000 (78%)
620 - StRS Median - Unif - 7442/10000 (74%)
621 - StRS TMean - Unif - 7347/10000 (73%)

Tuned parameters

*Horde a target class
628 - StRS Krum - Unif 6432/10000 (64%)
629 - StRS MKrum - Unif 6697/10000 (67%)
630 - StRS Median - Unif 6487/10000 (65%)
631 - StRS TMean - Unif 6818/10000 (68%)
"""

#Fine-tuning strata weights for different BR aggregation to influence to target stratum
#Krum
strata_params_dict = dict()
strata_params_dict['632'] = [1.0, 0.0, 0.0, 0.0]
strata_params_dict['633'] = [0.0, 1.0, 0.0, 0.0]
strata_params_dict['634'] = [0.0, 0.0, 1.0, 0.0]
strata_params_dict['635'] = [0.0, 0.0, 0.0, 1.0]

strata_params_dict2 = dict()
strata_params_dict2['636'] = [1.0, 1.0, 0.0, 0.0] #1-2
strata_params_dict2['637'] = [1.0, 0.0, 1.0, 0.0] #1-3
strata_params_dict2['638'] = [1.0, 0.0, 0.0, 1.0] #1-4
strata_params_dict2['639'] = [0.0, 1.0, 0.0, 1.0] #2-4
strata_params_dict2['640'] = [0.0, 0.0, 1.0, 1.0] #3-4

strata_params_dict3 = dict()
strata_params_dict3['641'] = [0.50, 0.05, 0.10, 0.35] #1-4-3-2
strata_params_dict3['642'] = [0.55, 0.05, 0.10, 0.30] #1-4-3-2
strata_params_dict3['643'] = [0.45, 0.05, 0.10, 0.40] #1-4-3-2
strata_params_dict3['644'] = [0.73, 0.05, 0.06, 0.16]

strata_params_dict4 = dict()
strata_params_dict4['645'] = [0.05, 0.25, 0.05, 0.65]
strata_params_dict4['646'] = [0.10, 0.20, 0.05, 0.65]

#Trimmed Mean
strata_params_dict_tmean = dict()
strata_params_dict_tmean['732'] = [1.0, 0.0, 0.0, 0.0]
strata_params_dict_tmean['733'] = [0.0, 1.0, 0.0, 0.0]
strata_params_dict_tmean['734'] = [0.0, 0.0, 1.0, 0.0]
strata_params_dict_tmean['735'] = [0.0, 0.0, 0.0, 1.0]

strata_params_dict2_tmean = dict()
strata_params_dict2_tmean['736'] = [0.5, 0.0, 0.0, 0.5] #4-1
strata_params_dict2_tmean['737'] = [0.0, 0.5, 0.0, 0.5] #4-2
strata_params_dict2_tmean['738'] = [0.0, 0.0, 0.5, 0.5] #4-3
strata_params_dict2_tmean['739'] = [0.5, 0.5, 0.0, 0.0] #1-2
strata_params_dict2_tmean['740'] = [0.5, 0.0, 0.5, 0.0] #1-3

#Median
strata_params_dict_median = dict()
strata_params_dict_median['832'] = [1.0, 0.0, 0.0, 0.0]
strata_params_dict_median['833'] = [0.0, 1.0, 0.0, 0.0]
strata_params_dict_median['834'] = [0.0, 0.0, 1.0, 0.0]
strata_params_dict_median['835'] = [0.0, 0.0, 0.0, 1.0]

strata_params_dict2_median = dict()
strata_params_dict2_median['836'] = [0.5, 0.0, 0.0, 0.5] #4-1
strata_params_dict2_median['837'] = [0.0, 0.5, 0.0, 0.5] #4-2
strata_params_dict2_median['838'] = [0.0, 0.0, 0.5, 0.5] #4-3
strata_params_dict2_median['839'] = [0.5, 0.5, 0.0, 0.0] #1-2
strata_params_dict2_median['840'] = [0.5, 0.0, 0.5, 0.0] #1-3

# strata_params = strata_params_dict2_median.copy()
# if __name__ == '__main__':
#     for exp in strata_params.keys():
#         print("Exp: {}. Strata weights: {}.".format(exp, strata_params[exp]))
#         START_EXP_IDX = int(exp)  # Change here
#         NUM_EXP = 1  # We can make this multiple experiments.
#
#         NUM_POISONED_WORKERS = 10  # The total number of poisoned workers/clients
#         REPLACEMENT_METHOD = replace_N_with_M
#         PARAMETERS_UPLOADED = 1.0
#         PARAMETERS_DOWNLOADED = 1.0  # set to 1.0 for unrestricted parameter sharing
#         KWARGS = {
#             "NUM_WORKERS_PER_ROUND": 25,  # 4, #25
#             "ASSUMED_POISONED_WORKERS_PER_ROUND": 5,  # an estimate of how many of the selected workers are byzantine
#             "NUM_KRUM": 3,  # Number of gradients to average. If NUM_KRUM = 1, basic Krum; if NUM_KRUM > 1, multi-Krum
#             "TRIM_PROPORTION": 1 / 5,
#             "SELECTION_STRATEGY": "Stratified_RS",  # sample selection: Simple_RS or Stratified_RS
#             "nlabels": 10,
#
#             # Stratified kwargs
#             "strata_weights": {'strata1': strata_params[exp][0], 'strata2': strata_params[exp][1], 'strata3': strata_params[exp][2], 'strata4': strata_params[exp][3]},
#             # stratum weights for weighted aggregation
#             "conceal_pois_class": True,  # Poisoned worker either conceals poisoned class
#             "influence": 'strata1',  # stratum to influence: "strata" + stratum number, default None
#             "stratify": 'median',  # stratification technique: 'median', 'kmeans'
#             "nclusters": 4,  # Number of clusters for the unsupervised stratification technique
#             "assumed_workers_per_round_stratum": int(25 / 4),  # We divide NUM_WORKERS_PER_ROUND by the number of clusters
#             "assumed_poisoned_per_round_stratum": int((10 / 50) * (25 / 4))
#             # Ratio of poisoned workers to num_workers applied to assumed workers per round
#
#         }
#         DATASET = "FashionMNIST"  #
#         TARGET = 6  # The target label as the poisoning/flipping
#         SOURCE = 4  # poisoned class
#         AGG = "StratMedian" #"StratKrum"  # "StratFedAvg" #MultiKrum" #"StratTrimMean" #"StratKrum" #Change here
#         INIT = "Randomized"  # "Default"
#         DISTRIBUTION = "Non-IID_v2"
#
#         count = 0
#         for experiment_id in range(START_EXP_IDX, START_EXP_IDX + NUM_EXP):
#             count += 1  # experiment count
#             rd.seed(1234 + 100 * count)
#             np.random.seed(1234 + 100 * (count - 1))
#             with open("experiment_param_notes.txt", 'a') as f:
#                 f.write(
#                     f"{experiment_id}, {DATASET}, {NUM_POISONED_WORKERS}, {SOURCE}, {TARGET}, {get_class_label_from_num(DATASET, SOURCE)}, {get_class_label_from_num(DATASET, TARGET)}, {PARAMETERS_DOWNLOADED}, {PARAMETERS_UPLOADED}, {AGG}, {INIT}, {DISTRIBUTION}\n")
#             print(f"Exp ID: {experiment_id}, Num pois: {NUM_POISONED_WORKERS}, Uploaded: {PARAMETERS_UPLOADED}")
#             run_exp(REPLACEMENT_METHOD, NUM_POISONED_WORKERS, KWARGS, RandomSelectionStrategy(), experiment_id,
#                     PARAMETERS_UPLOADED, PARAMETERS_DOWNLOADED, DATASET, INIT, DISTRIBUTION, SOURCE, TARGET, AGG)  # Edited: I added TARGET as a parameter

###RUN A SCRIPT FOR THE CIFAR10

#Tuned StMKrum: {'strata1': 0.50, 'strata2': 0.05, 'strata3': 0.10, 'strata4': 0.35}
#Tuned StTMean:
#Tuned StMedian:

"""
Stratified TMean, FMNIST 1 to 3
854 - [1, 0, 0, 0] #Top 3
855 - [0, 1, 0, 0] #Top 2
856 - [0, 0, 1, 0] 
857 - [0, 0, 0, 1] #Top 1 but wavy

858 - [0.5, 0.5, 0, 0]
859 - [0, 0.5, 0.5, 0]
860 - [0, 0.5, 0, 0.5] 7402/10000 (74%) If s4 up, more wavy

861 - [0.0, 0.20, 0.0, 0.80] 7456/10000 (75%), 7452/10000 (75%), 7362/10000 (74%), 7470/10000 (75%)

"""

if __name__ == '__main__':
    START_EXP_IDX = 454 #Change here
    NUM_EXP = 1 #We can make this multiple experiments.

    NUM_POISONED_WORKERS = 10 #The total number of poisoned workers/clients
    REPLACEMENT_METHOD = replace_N_with_M
    PARAMETERS_UPLOADED = 1.0
    PARAMETERS_DOWNLOADED = 1.0 #set to 1.0 for unrestricted parameter sharing
    KWARGS = {
        "NUM_WORKERS_PER_ROUND" : 25, #4, #25
        "ASSUMED_POISONED_WORKERS_PER_ROUND" : 5, #an estimate of how many of the selected workers are byzantine
        "NUM_KRUM" : 3, #Number of gradients to average. If NUM_KRUM = 1, basic Krum; if NUM_KRUM > 1, multi-Krum
        "TRIM_PROPORTION": 1/5,
        "SELECTION_STRATEGY": "Simple_RS", #sample selection: Simple_RS or Stratified_RS
        "nlabels": 10,

        #Stratified kwargs
        "strata_weights": {'strata1': 0.25, 'strata2': 0.25, 'strata3': 0.25, 'strata4': 0.25}, #stratum weights for weighted aggregation
        "conceal_pois_class": True, #Poisoned worker either conceals poisoned class
        "influence": None, #stratum to influence: "strata" + stratum number, default None
        "stratify": 'median', #stratification technique: 'median', 'kmeans'
        "nclusters": 4, #Number of clusters for the unsupervised stratification technique
        "assumed_workers_per_round_stratum": int(25/4), #We divide NUM_WORKERS_PER_ROUND by the number of clusters
        "assumed_poisoned_per_round_stratum": int((10/50)*(25/4)) #Ratio of poisoned workers to num_workers applied to assumed workers per round

    }
    DATASET = "FashionMNIST" #
    TARGET = 6 #The target label as the poisoning/flipping
    SOURCE = 4 #poisoned class
    AGG = "FedAvg" #"StratKrum" #StratKrum" #"StratFedAvg" #MultiKrum" #"StratTrimMean" #"StratKrum" #Change here
    INIT= "Default" # "Default"
    DISTRIBUTION = "Non-IID"

    count = 0
    for experiment_id in range(START_EXP_IDX, START_EXP_IDX + NUM_EXP):
        count += 1 #experiment count
        rd.seed(1234 + 100 * count)
        np.random.seed(1234 + 100 * (count-1))
        with open("experiment_param_notes.txt", 'a') as f:
            f.write(f"{experiment_id}, {DATASET}, {NUM_POISONED_WORKERS}, {SOURCE}, {TARGET}, {get_class_label_from_num(DATASET, SOURCE)}, {get_class_label_from_num(DATASET, TARGET)}, {PARAMETERS_DOWNLOADED}, {PARAMETERS_UPLOADED}, {AGG}, {INIT}, {DISTRIBUTION}, {KWARGS['strata_weights'], KWARGS['conceal_pois_class'], KWARGS['influence'], KWARGS['stratify'], KWARGS['nclusters']}\n")
        print(f"Exp ID: {experiment_id}, Num pois: {NUM_POISONED_WORKERS}, Uploaded: {PARAMETERS_UPLOADED}")
        run_exp(REPLACEMENT_METHOD, NUM_POISONED_WORKERS, KWARGS, RandomSelectionStrategy(), experiment_id, PARAMETERS_UPLOADED, PARAMETERS_DOWNLOADED, DATASET, INIT, DISTRIBUTION, SOURCE, TARGET, AGG) #Edited: I added TARGET as a parameter

    #This uses a simple randoom selection of workers using RandomSelectionStrategy()