# DP-SCD 

Author: Georgios Damaskinos (georgios.damaskinos@gmail.com)

_DP-SCD_ is a privacy-preserving optimization algorithm introduced in [Differentially Private Stochastic Coordinate Descent](https://arxiv.org/pdf/2006.07272.pdf).

## Requirements
* Python (3.5+)
* ```pip install sklearn pandas seaborn wget sympy```

## Quick-start

The following reproduces a single (out of the total of 10 used) run for the result shown in Figure 1(a) (DP-SCD (ε=0.1)).

```
python driver.py --app RR --solver PSCD --max_iter 50 --lambda 1e-4 --eps 0.1 --lot_ratio 1000 --C 0.1 --dataset msd --dual --valid_size 0.25 --seed 1
```

## Components

* [Preprocessor](preprocessor.py): Loads the data and handles preprocessing. Information on how to obtain the data is embedded to the code.

* [Optimizer](optimizer.py): Implements the various optimization algorithms, namely SCD, SGD, DP-SCD, DP-SGD, in an application-agnostic manner.

* Applications ([ridge regression](standaloneRR.py), [logistic regression](standaloneLR.py), [SVMs](standaloneSVM.py)): Implement the application-specific code required by the Optimizer to perform training.

* [Driver](driver.py): Tool for parallel (single machine) training and evaluation for a given set of hyperparameters.

* [Accountant](accountant.py): Tool for measuring the privacy loss based on the moments accountant.
