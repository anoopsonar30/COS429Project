# COS 429 Final Project - Anoopkumar Sonar and Sahan Paliskara
Code for the paper:
Towards Fairness in Visual Recognition: Effective Strategies for Bias Mitigation - https://github.com/princetonvisualai/DomainBiasMitigation

Invariant Risk Minimization - https://github.com/facebookresearch/InvariantRiskMinimization


## Requirements
* Python 3.6+
* PyTorch 1.0+
* h5py
* tensorboardX

## Data Preparation
First download and unzip the CIFAR-10 and CINIC-10 by running the script `download.sh` in DBM
Run the `preprocess_data.py` to generate data for all experiments.

## Run Experiments
To conduct experiments, run `main.py` with corresponding arguments (`experiment` specifies which experiment to run, `experiment_name` specifies a name to this experiment for saving the model and result). For example:

```
## Switch to a different branch
git checkout AnoopClone
## Launch Training
cd DBM
python main.py --experiment cifar_domain_independent --experiment_name domainIndie1 --random_seed 1
## Check the records directory for accuracy results
cd record
## and use RBA to further reduce bias
cd RBA
jupyter notebook RBA_cifar.ipynb
```

After running, the experiment result will be saved under `record/experiment/experiment_name`
