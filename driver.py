"""
  Copyright (c) 2020 Georgios Damaskinos
  All rights reserved.
  @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
  This source code is licensed under the MIT license found in the
  LICENSE file in the root directory of this source tree.
"""

import os
import multiprocessing
import argparse
import time
import sys
from os.path import expanduser
home = expanduser("~")
import itertools
from functools import partial
import subprocess
import pandas as pd
import socket

parser = argparse.ArgumentParser()
parser.add_argument("--lambda", type=float, nargs='+',
    default=[1.0], action='store')
parser.add_argument("--sigma", type=float, nargs='+',
    default=[0.0], action='store')
parser.add_argument("--eps", type=float, nargs='+',
    default=[0.0], action='store')
parser.add_argument("--C", type=float, nargs='+',
    default=[0.0], action='store')
parser.add_argument("--sample_ratio", type=float, nargs='+',
    default=[1.0], action='store')
parser.add_argument("--lot_ratio", type=float, nargs='+',
    default=[1.0], action='store')
parser.add_argument("--max_iter", type=int, nargs='+',
    default=[10], action='store')
parser.add_argument("--out_iter", type=int, nargs='+',
    default=[1], action='store')
parser.add_argument("--K", type=int, nargs='+',
    default=[1], action='store')
parser.add_argument("--gamma", type=float, nargs='+',
    default=[1.0], action='store')
parser.add_argument("--seed", type=int, nargs='+',
    default=[1], action='store')
parser.add_argument('--dual', action='store_true')
parser.add_argument("--app", type=str, default='RR', action='store')
parser.add_argument("--solver", type=str, default='SCD', action='store')
parser.add_argument("--dataset", type=str, default='msd', action='store')
parser.add_argument('--valid_size', type=float, default=0, action='store')
parser.add_argument('--pool_size', type=int, action='store',
    help='Maximum number of parallel processes to spawn')
parser.add_argument("--outputPrefix", type=str, action='store',
    help='Output prefix for multi process output logs; if None prints to stdout')

# preprocess args before imports to set parallelism
args = sys.argv[1:]
print("Executing on: %s" % socket.gethostname())
print("Reproduce with:\n```python " + " ".join(sys.argv) + "```")
args = parser.parse_args(args)

param_names = ['lambda', 'sigma', 'eps', 'C', 'sample_ratio',
               'lot_ratio', 'max_iter', 'out_iter', 'K', 'gamma', 'seed']

# param combinations
param_combs = list(itertools.product(*map(lambda param: vars(args)[param], param_names)))

# each row holds the hyperparameters for 1 run
run_params = pd.DataFrame(columns=param_names)
for run_idx in range(len(param_combs)):
  run_params.loc[run_idx] = param_combs[run_idx]

param_types = {}
for param in param_names:
  if type(vars(args)[param][0]) == int:
    param_types[param] = 'int32'
  else:
    param_types[param] = 'float'
run_params = run_params.astype(param_types)

if not args.pool_size is None:
  pool_size = args.pool_size
else:
  pool_size = len(run_params)

# define parallelism for each process
total_cores = multiprocessing.cpu_count()
threads = max(1, int(total_cores/pool_size - 2))
print("Num threads: %g" % threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
os.environ["OMP_NUM_THREADS"] = str(threads)
os.environ["MKL_NUM_THREADS"] = str(threads)

import numpy as np
np.random.seed(1)

from sklearn import linear_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, roc_curve, auc, log_loss, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import csv
import scipy
import pickle
from inspect import signature

import preprocessor
import standaloneLR
import standaloneRR
import standaloneSVM
import accountant


def parallelTrainEval(run_params, dual, app, solver, valid_size, start_time, outputPrefix=None, jobID=0):
  """Wrapper around trainEval for parallel execution
  Must have globally defined: X_train, Y_train, X_test, Y_test
  Args:
    run_params.iloc[jobID]: parameter values for this job
    ...
    outputLogs (list): output file for logging for each job
  """
  print("STARTING thread: %d" % jobID, flush=True)

  prev_stdout = sys.stdout
  if outputPrefix is not None:
    outputFilename = outputPrefix
    for col in run_params.columns:
      outputFilename += col[:2] + "{:g}".format(run_params[col].iloc[jobID]) + '_'
    outputFilename = outputFilename[:-1]

    if os.path.exists(outputFilename):
      with open(outputFilename) as f:
        txt = f.read()
        if ('Training complete!' in txt) or ('Stopped training' in txt):
          # log exists - skip
          print("SKIPPING...\nFINISHING thread: %d / %d" % (jobID, len(run_params)), flush=True)
          return

    sys.stdout = open(outputFilename, "w")

  try:
    ret = trainEval(X_train, Y_train, X_test, Y_test, valid_size=valid_size,
                    lambda_=run_params['lambda'].iloc[jobID],
                    sigma=run_params['sigma'].iloc[jobID],
                    eps=run_params['eps'].iloc[jobID],
                    C=run_params['C'].iloc[jobID],
                    sample_ratio=run_params['sample_ratio'].iloc[jobID],
                    lot_ratio=run_params['lot_ratio'].iloc[jobID],
                    max_iter=run_params['max_iter'].iloc[jobID],
                    out_iter=run_params['out_iter'].iloc[jobID],
                    K=run_params['K'].iloc[jobID],
                    gamma=run_params['gamma'].iloc[jobID],
                    seed=run_params['seed'].iloc[jobID],
                    dual=dual, app=app, solver=solver)

  except ArithmeticError as e:
    print("Stopped training: ", str(e), flush=True)

  sys.stdout = prev_stdout
  print("FINISHING thread: %d / %d\tElapsed time: %g secs" % (
    jobID, len(run_params), time.time()-start_time), flush=True)

  return


def trainEval(X_train, Y_train, X_test, Y_test, valid_size=0, lambda_=1.0,
              sigma=0, eps=0, sample_ratio=1, lot_ratio=1, C=0,
              dual=True, app='RR', solver='SCD', K=1, gamma=1, out_iter=1,
              max_iter=10, sigmaP=None, verbose=False, seed=1):
  """Training and evaluation procedure for a hardcoded optimizer
  Args:
    valid_size (float): size (percentage of the training size) for the
      validation set. If 0 => validation set = test set
    lambda_ (float): Regularization strength. It must be a positive float.
      Larger regularization values imply stronger regularization.
    app (str): 'RR', 'LR', 'SVM'
    solver (str): see optimizer.py#__init__
    see standaloneRR for the rest
  Returns:
    return of solver.fit()
    performance measurements (e.g., MAE for regression, Accuracy for classification)
  """
  np.random.seed(seed)


  """Solver"""
  # standalone solver
  if app == 'LR':
    optimizer = standaloneLR.LogisticRegression
  elif app == 'RR':
    optimizer = standaloneRR.RidgeRegression
  elif app == 'SVM':
    optimizer = standaloneSVM.SVM
  else:
    raise NotImplementedError("Unknown app")

  solver = optimizer(fit_intercept=False, seed=seed,
        dual=dual, verbose=verbose, sigma=sigma, eps=eps, C=C, regularizer=lambda_,
        sample_ratio=sample_ratio, lot_ratio=lot_ratio, solver=solver, gamma=gamma,
        K=K, out_iter=out_iter, max_iter=max_iter, sigmaP=sigmaP)

  """Training"""

  print("Valid_size = %g" % valid_size, flush=True)
  if valid_size == 0:
    X_val = X_test
    Y_val = Y_test
  else:
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train,
        test_size=valid_size)

  print("Fitting for (train, validation) shapes: (%s, %s)" % (
    X_train.shape, X_val.shape), flush=True)
  t0 = time.time()
  if len(signature(solver.fit).parameters) == 4:
    # monitor cost+performance (!only for standalone solvers)
    ret = solver.fit(X_train, Y_train, X_val, Y_val)
  else:
    ret = solver.fit(X_train, Y_train) # monitor only training cost
  print("Training time (s):  {0:.2f}".format(time.time()-t0))

  """Evaluation"""
  solver.evaluate(X_test, Y_test)

  print("Training complete!", flush=True)

  return ret

def main(args):

  print("Loading data...", flush=True)
  global X_train, Y_train, X_test, Y_test
  X_train, Y_train, X_test, Y_test = preprocessor.load(args.dataset)

  print('X_train:',X_train.shape)
  print('Y_train:',Y_train.shape)
  print('X_test:',X_test.shape)
  print('Y_test:',Y_test.shape, flush=True)

  # hyper-parameter search with multiprocessing
  jobIDs = np.array(range(0, len(run_params)))
  part = partial(parallelTrainEval, run_params, args.dual, args.app, args.solver,
                 args.valid_size, time.time(), args.outputPrefix)

  pool = multiprocessing.Pool(args.pool_size)
  pool.map(part, jobIDs)
  pool.close()
  pool.join()
  print("EXECUTION DONE", flush=True)

if __name__ == "__main__":
  main(args)
