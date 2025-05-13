# Introduction
This repository contains the code and experiments for our project in CSC6022 at CUHK-SZ. We implement and evaluate six tuning-free optimization methods on various neural network architectures and datasets.

# Methods
We incorporate the following tuning-free schedules, drawing from their official implementations:

- **COCOB:** [parameterfree](https://github.com/bremen79/parameterfree)  
- **DoG / L-DoG:** [dog](https://github.com/formll/dog)  
- **D-Adaptation:** [dadaptation](https://github.com/facebookresearch/dadaptation)  
- **Schedule-Free:** [schedulefree](https://github.com/facebookresearch/schedule_free)  
- **Prodigy:** [prodigyopt](https://github.com/konstmish/prodigy)

# Files and Directories
- **`py`**: Contains the network architectures sourced from [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar).  
- **`outs`**: Stores all experiment outputs.  
- **`pic`**: Contains all figures generated during the project. The notebook `Figure Repetition.ipynb` can be used to reproduce these plots.  
- **`run.py`**: Provides a simple demonstration of training a ResNet18 on CIFAR-10 using different tuning-free optimizers.

We hope this repository assists you in exploring and understanding the tuning-free schedules. Enjoy experimenting!