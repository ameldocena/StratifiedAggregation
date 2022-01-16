# DataPoisoning

## Setup

Before running any of the experiments, you must run the code to generate the default models and data distribution. 
These scripts are unchanged from the source repository.
```
python3 generate_data_distribution.py
python3 generate_default_models.py
```

## Hyperparameters

The hyperparameters used for training are the same as those proposed by the original work and are reproduced below.

CIFAR10:
- Batch size: 10
- LR: 0.01
- Number of epochs: 200
- Momentum: 0.5
- Scheduler step size: 50
- Scheduler gamma: 0.5
- Min_lr: 1e-10

Fashion-MNIST:
- Batch size: 4
- LR: 0.001
- Number of epochs: 200
- Momentum: 0.9
- Scheduler step size: 10
- Scheduler gamma: 0.5
- Min_lr: 1e-10

## Training

The main file used for training can be run as follows:
```
python3 run_federated_training.py
```

The experiment settings can be adjusted in this file, but adding a CLI or at least modifying the script to take command line arguments is high priority.

The two core functions to run the experiments are defined in server_core.py

The aggregators module includes a base Aggregator class that shows generally how aggregation should work. 
This base class implements the most basic form of FedAvg which only works with complete parameter sharing, but the fedavg.py module extends it to work in restricted cases.
The aligned_avg.py module defines an AlignedAvg aggregator that will be able to use 0 imputation as well as filter-wise or layer-wise mean imputation to fill missing values before aligning filters with either the EMD from the Optimal Transport paper or the Hungarian Algorithm (scipy.optimize linear_sum_assignment) used by PFNM. 
The get_alignment function has not been updated yet, but I will push the changes later this week.

## TODO - Immediate Next Steps

- Add pip/conda requirements
- Improve documentation, update all docstrings
- Push changes to aligned_avg.py
- Figure out best way to share results
- Add CLI for run_federated_training.py
- Bash wrapper to run multiple experiments without storage bottlenecks

## Citation
I have tried to leave comments throughout the code that indicates if a function was present in the original repository or is something that I added, but these have not been updated since reorganization.
The attribution requested by the authors of the source code is included below.
```
@ARTICLE{2020arXiv200708432T,
       author = {{Tolpegin}, Vale and {Truex}, Stacey and {Emre Gursoy}, Mehmet and
         {Liu}, Ling},
        title = "{Data Poisoning Attacks Against Federated Learning Systems}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Machine Learning, Computer Science - Cryptography and Security, Statistics - Machine Learning},
         year = 2020,
        month = jul,
          eid = {arXiv:2007.08432},
        pages = {arXiv:2007.08432},
archivePrefix = {arXiv},
       eprint = {2007.08432},
 primaryClass = {cs.LG},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020arXiv200708432T},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
