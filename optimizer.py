"""
  Copyright (c) 2020 Georgios Damaskinos
  All rights reserved.
  @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
  This source code is licensed under the MIT license found in the
  LICENSE file in the root directory of this source tree.
"""

import numpy as np
import pandas as pd


import builtins as __builtin__
from abc import ABC, abstractmethod
import time
import sys
import os
import scipy
import sympy
import math
from sklearn.metrics import mean_squared_error, r2_score, pairwise
import sklearn
import scipy

import accountant

def print(*args, **kwargs):
  return __builtin__.print(*args, flush=True, **kwargs)

class Optimizer(ABC):
  """Optimizer for training linear models
  Implementation is based on the equations in https://arxiv.org/pdf/1702.07005.pdf[paper]
  there are some differences noted on the code
  """

  def __init__(self, max_iter=1, regularizer=1.0, dual=True, sigma=0, eps=0,
      sample_ratio=1, lot_ratio=1, C=0, verbose=False, fit_intercept=True,
      solver='SCD', K=1, out_iter=1, gamma=1, sigmaP=None, seed=1):
    """
    Args:
      sigma (float): float controlling the noise addition
        a value of 0 indicates no noise addition
      eps (float): float setting the privacy bound
        this must be given ONLY with sigma=0 => sigma will be determined automatically
        a value of 0 deactivates this param
      sample_ratio (float): sample_size/dataset_size
        if 0 => sample_size = 1
        if 0 < sample_ratio <= 1 => sample_size = sample_ratio * N
        if sample_ratio > 1 => sample_size = sample_ratio
        Primal: Total_updates = M * max_iter, each based on a sample with size: sample_size
        Dual:   Total_updates = sample_size * max_iter
      lot_ratio (float): lot_size/dataset_size
        if 0 => lot_size = 1
        Primal: if 0 < lot_ratio <= 1 => lot_size = lot_ratio * M
        Dual:   if 0 < lot_ratio <= 1 => lot_size = lot_ratio * N
        if lot_ratio > 1 => lot_size = lot_ratio
      max_iter (float): defines (along with sample_ratio) the total_updates
        Primal: Total_updates = M * max_iter
        Dual:   Total_updates = sample_size * max_iter
      C (float): norm bounds (use only one C for sgd)
        default: 0 => no scaling
        -1 => C = np.inf
        if C > 0:
          noise is scaled to sigma * C
      solver (str): 'SCD', 'SGD', 'COCOA'
      K (int): number of machines for COCOA solver
      out_iter (int): number of COCOA solver iterations
      gamma (float): scaling factor for updates
        learning rate for SGD solver
        aggregation parameter for COCOA+ solver:
          for summing COCOA:  gamma = 1
          for averaging COCOA: gamma = 1/K
      sigmaP (float): local subproblem parameter for COCOA+ solver
        if None: sigmaP = K
        for averaging COCOA: sigmaP = 1
      verbose (boolean): Prints various info for training including
        hyperparameter values and performance per epoch
    """

    np.random.seed(seed)

    self.max_iter = max_iter
    self.lambda_ = regularizer
    self.verbose = verbose
    self.dual = dual
    self.sigma = sigma
    self.eps = eps
    self.sample_ratio = sample_ratio
    self.lot_ratio = lot_ratio
    if C == -1:
      self.C = np.inf
    else:
      self.C = C
    self.fit_intercept = fit_intercept
    self.solver = solver
    self.K = K
    self.out_iter = out_iter
    self.gamma = gamma
    if sigmaP is None:
      sigmaP = K
    self.sigmaP = sigmaP
    print("Seed: %g" % seed)
    print("Max_iter: %g" % self.max_iter)
    print("Fit_intercept: %s" % self.fit_intercept)
    print("Lambda: %g" % self.lambda_)
    print("Sample_ratio: %g" % self.sample_ratio)
    print("Lot_ratio: %g" % self.lot_ratio)
    if self.eps > 0:
      print("Target eps: %g" % self.eps)
    else:
      print("Sigma: %g" % self.sigma)
    print("C: %g" % self.C)
    print("Solver: %s" % self.solver)
    print("Gamma: %g" % self.gamma)
    print("SigmaP: %g" % self.sigmaP)
    if self.solver == 'SSCD' or self.solver == 'PSCD':
      print("Dual: %s" % self.dual)
    elif self.solver == 'COCOA':
      print("K: %d" % self.K)
      print("Out_iter: %d" % self.out_iter)

  def _vecCompare(self, v1, v2):
    print("Cosine similarity: %g" %
        pairwise.cosine_similarity(v1.reshape(1,-1), v2.reshape(1, -1))[0][0])
    print("Duality norm1: %g" % np.linalg.norm(v1))
    print("Duality norm2: %g" % np.linalg.norm(v2))
    print("Euclidean distance: %g" %
     pairwise.euclidean_distances(v1.reshape(1,-1), v2.reshape(1, -1))[0][0])

  def _privacy_accounting(self, q, iters, eps, delta=0.001):
    """Compute sigma if necessary and use it to compute eps"""

    acc = accountant.GaussianMomentsAccountant(verbose=False,
                                               moment_orders=128, # for eps=0.1
                                               seed=np.random.get_state()[1][0])
    if eps > 0:
      self.sigma = acc.compute_sigma2(eps=eps, delta=delta, q=q,
                                     iters=iters)
      print("Sigma: %g" % self.sigma)
    if self.sigma > 0:
      acc.accumulate_privacy_spending(sigma=self.sigma, q=q,
                                      iters=iters, reset=True)
      measured_eps = acc.get_privacy_spent(target_deltas=[delta])[0][0]
      if abs(measured_eps - eps) > eps:
        raise(ArithmeticError("Target eps=%g whereas measured eps=%g" % (
        eps, measured_eps)))
      print("Eps: %g" % acc.get_privacy_spent(target_deltas=[delta])[0][0])
      print("Delta: %g" % delta)



  @abstractmethod
  def _dual_loss(self, X, y, alpha, w):
    pass

  @abstractmethod
  def _primal_loss(self, X, y, theta):
    pass

  @abstractmethod
  def _dual_optimal_dampening(self, y, alpha, w, delta_alpha, delta_w, sigma, lambda_):
    pass

  @abstractmethod
  def _primal_optimal_dampening(self, y, theta, w, delta_theta, delta_w, sigma, lambda_):
    pass

  def _init_alpha(self, y):
    N = y.shape[0]
    return np.zeros(N)

  @abstractmethod
  def _delta_alpha(self, yn, w, Xn, alpha_n, lambda_, sigmaP, N):
    pass

  @abstractmethod
  def _delta_theta(self, yn, w, Xn, alpha_n, lambda_, sigmaP, N):
    pass

  @abstractmethod
  def _gradient(self, yn, w, Xn):
    pass

  @abstractmethod
  def predict(self, X_test):
    pass

  @abstractmethod
  def evaluate(self, X_test, Y_test, epoch, time, cost=-1):
    pass

  def objective(self, X, y, theta, alpha, lambda_):
    """Summation of regularization term and application-specific loss"""
    N = X.shape[0]
    primal_loss = self._primal_loss(X, y, theta)
    primal_objective = primal_loss + lambda_ / 2 * np.linalg.norm(theta)**2
    if self.verbose and not alpha is None:
      dual_loss = self._dual_loss(X, y, alpha)
      dual_objective = dual_loss + 1/(2*lambda_*N**2) * np.linalg.norm(X.T.dot(alpha)) ** 2
      # see cd thesis equation 2.12
      duality_gap = theta.T.dot(X.T).dot(alpha) + primal_loss + dual_loss
      print("Primal loss: %g" % primal_loss)
      print("Primal objective: %g" % primal_objective)
      print("Dual loss: %g" % dual_loss)
      print("Dual objective: %g" % dual_objective)
      print("Duality gap: %g" % abs(primal_objective + dual_objective))
    return primal_objective

  def dual_PSCD(self, X, y, X_test, Y_test, alpha_init=None, w_init=None):
    """Dual Stochastic Coordinate Descent with independent updates per lot
    Args:
      alpha_init, w_init (np.array): initial vectors. Useful for COCOA
    """

    start_time = time.time()

    N, M = X.shape

    # initialize dual and shared vector
    if not alpha_init is None:
      alpha = alpha_init.copy()
    else:
      alpha = self._init_alpha(y)

    if not w_init is None:
      w = w_init.copy()
    else:
      if (isinstance(X, scipy.sparse.csr_matrix)):
        w = X.T.dot(alpha).reshape(-1,1) # dual shared vector
      else:
        w = X.T.dot(alpha)

    # Output (DP) mechanism M_k (i.e., primal model):
    # M_k = 1/lambda (\sum_{i \in lot_k} delta_w + N(0, sigma^2))
    self.coef_ = w / (self.lambda_ * N)

    perm = np.arange(N)

    # derive lot_size, sample_size and scale lot_ratio to [0,1]
    if self.lot_ratio <= 1:
      lot_size = max(1, math.ceil(self.lot_ratio * N))
    else:
      lot_size = self.lot_ratio
      self.lot_ratio /= N

    # scale the total_updates such that the last update includes a lot processing
    total_updates = int(N * self.max_iter)
    if lot_size > total_updates:
      print("Truncating lot_size...")
      lot_size = total_updates
    total_updates -= int(total_updates % lot_size)

    print("Lot_size: %d" % lot_size)
    print("Total_updates: %d" % total_updates)

    delta_alpha = np.zeros(N)
    delta_w = np.zeros(M)
    epoch = -1
    prevEvalEpoch = -1

    # list of privacy params for the output mechanism M = M_a + M_b + ...
    # each entry contains a (iterative) mechanism Mi (see accountant.py)
    # in this case: M = M_a = M_1 + M_2 + ... + M_k
    self.iters = [int(total_updates / lot_size)] # compositions for the updates
    self.q = min([1], [lot_size/(N*self.K)]) # sampling probability for the updates
    print("Privacy params (q, iters): (%s, %s)" % (self.q, self.iters))
    self._privacy_accounting(eps=self.eps, q=self.q[0], iters=self.iters[0])
                             #delta=1/(N*self.K)**2)

    lot = []

    # TODO: edit for manual scaling OR constant dampening OR optimal dampening
    # manual scaling schedule
    a_warmup_thres = 2000
    a_warmup_thres = -1 # deactivate
    w_warmup_thres = 2000
    w_warmup_thres = -1 # deactivate
    a_warmup_scale = 3000
    w_warmup_scale = 100

    # constant dampening (by self.gamma)
    constant_dampen_thres = np.inf # always constant dampening
    #constant_dampen_thres = -1 # deactivate => always optimal dampening

    # averaging
    #self.sigmaP = 1
    #self.gamma = 1/lot_size

    # summing
    self.sigmaP = lot_size
    self.gamma = 1

    print("sigmaP: %g\ngamma: %g" % (self.sigmaP, self.gamma))

    for i in range(total_updates):
      # completed one epoch
      if i % N == 0:
        np.random.shuffle(perm)
        epoch += 1

      # evaluate after lot processing
      if (lot_size > N and i % N ==0) or (
      epoch > prevEvalEpoch and i % (int(N / lot_size) * lot_size) == 0):
        prevEvalEpoch = epoch

        self.coef_ = w / (self.lambda_ * N)

        cost = self.objective(X, y, self.coef_, alpha, self.lambda_)
        self.evaluate(X_test, Y_test, epoch, (time.time() - start_time)*1000, cost)
        if self.verbose:
          self._vecCompare(X.T.dot(alpha), w)
          if lot_size <= N and epoch > 0:
            assert process_lot_i == i - 1
            print("Median delta_alpha norm: %g" % np.median(delta_alpha_norms))
            print("Median delta_w norm: %g" % np.median(delta_w_norms))
            if gammas != []:
              print("Average gamma: %g" % np.mean(gammas))
              print("Std gamma: %g" % np.std(gammas))

        gammas = []
        deltas = []
        delta_w_norms = []
        delta_alpha_norms = []
        numLots = 0

      # compute delta alpha
      n = perm[i % N]
      lot.append(n)

      delta_alpha[n] = self._delta_alpha(y[n], w, X[n,:], alpha[n],
                                        self.lambda_, self.sigmaP, N)

      # scale delta alpha
      if self.C > 0:
        scale = np.maximum(1, np.abs(delta_alpha[n]) / self.C)
        delta_alpha[n] /= scale

      # compute delta w
      if (isinstance(X, scipy.sparse.csr_matrix)):
        delta_w += X[n,:].multiply(delta_alpha[n]).transpose()
      else:
        delta_w += X[n,:]*delta_alpha[n]

      # Fast path
      if self.sigma == 0:
        w += delta_w
        delta_w = np.zeros(M)
      if self.C == 0 and lot_size == 1 and self.sigma == 0:
        alpha[lot] += delta_alpha[lot]
        numLots += 1
        self.coef_ = w / self.lambda_ # primal model
        process_lot_i = i
        lot = [] # reset lot

      # process one lot
      elif (i+1) % lot_size == 0:

        # dampen delta values (out-loop)
        if epoch < constant_dampen_thres:
          gamma = self.gamma
        else:
          gamma = self._dual_optimal_dampening(y[lot], alpha[lot], w, delta_alpha[lot], delta_w,
            self.sigma, self.lambda_) # must use with C=np.inf

        if self.verbose:
          gammas.append(gamma)
          delta_alpha_norm = np.linalg.norm(delta_alpha[lot])
          delta_alpha_norms.append(delta_alpha_norm)
          #med_alpha = np.median(delta_alpha_norms)

          delta_w_norm = np.linalg.norm(delta_w)
          delta_w_norms.append(delta_w_norm)
          #med_w = np.median(delta_w_norms)

        # perturb delta alpha and delta w
        w_noise = None
        if self.sigma > 0:
          a_noise = np.random.normal(0, self.sigma * np.sqrt(2*self.C**2), size=len(lot))
          w_noise = np.random.normal(0, self.sigma * np.sqrt(2*self.C**2), size=M)
          if (isinstance(X, scipy.sparse.csr_matrix)):
            w_noise = w_noise.reshape(-1, 1)
          delta_alpha[lot] += a_noise
          delta_w += w_noise

        # update
        w += gamma * delta_w
        alpha[lot] += gamma * delta_alpha[lot]


        numLots += 1
        self.coef_ = w / (self.lambda_ * N) # primal model
        process_lot_i = i
        delta_w = np.zeros(M)
        lot = [] # reset lot

    assert process_lot_i == i # output model is perturbed (if DP)

    cost = self.objective(X, y, self.coef_, alpha, self.lambda_)
    self.evaluate(X_test, Y_test, epoch+1, (time.time() - start_time)*1000,
                  cost, lastTrainingEpoch=True)

    # output useful for COCOA
    if not alpha_init is None and not w_init is None:
      self._delta_alpha_k = alpha - alpha_init
      self._delta_w = (w - w_init) / self.sigmaP

    return self


  def dual_SSCD(self, X, y, X_test, Y_test, alpha_init=None, w_init=None):
    """Dual Stochastic Coordinate Descent with sequencial inter-dependent updates
    Args:
      alpha_init, w_init (np.array): initial vectors. Useful for COCOA
    """

    start_time = time.time()

    N, M = X.shape

    # initialize dual and shared vector
    if not alpha_init is None:
      alpha = alpha_init.copy()
    else:
      alpha = self._init_alpha(y)

    if not w_init is None:
      w = w_init.copy()
    else:
      if (isinstance(X, scipy.sparse.csr_matrix)):
        w = X.T.dot(alpha).reshape(-1,1) # dual shared vector
      else:
        w = X.T.dot(alpha)

    # Output (DP) mechanism M_k (i.e., primal model):
    # M_k = 1/lambda (\sum_{i \in lot_k} delta_w + N(0, sigma^2))
    self.coef_ = w / (self.lambda_ * N)

    perm = np.arange(N)

    # derive lot_size, sample_size and scale lot_ratio to [0,1]
    if self.lot_ratio <= 1:
      lot_size = max(1, math.ceil(self.lot_ratio * N))
    else:
      lot_size = self.lot_ratio
      self.lot_ratio /= N

    # scale the total_updates such that the last update includes a lot processing
    total_updates = int(N * self.max_iter)
    if lot_size > total_updates:
      print("Truncating lot_size...")
      lot_size = total_updates
    total_updates -= int(total_updates % lot_size)

    print("Lot_size: %d" % lot_size)
    print("Total_updates: %d" % total_updates)

    epoch = -1
    prevEvalEpoch = -1

    # list of privacy params for the output mechanism M = M_a + M_b + ...
    # each entry contains a (iterative) mechanism Mi (see accountant.py)
    # in this case: M = M_a = M_1 + M_2 + ... + M_k
    self.iters = [self.K * self.out_iter * int(total_updates / lot_size)] # compositions for the updates
    self.q = min([1], [lot_size/(N*self.K)]) # sampling probability for the updates
    print("Privacy params (q, iters): (%s, %s)" % (self.q, self.iters))
    self._privacy_accounting(eps=self.eps, q=self.q[0], iters=self.iters[0])
                             #delta=1/(N*self.K)**2)

    lot = []

    for i in range(total_updates):
      # completed one epoch
      if i % N == 0:
        np.random.shuffle(perm)
        epoch += 1

      # evaluate after lot processing
      if (lot_size > N and i % N ==0) or (
      epoch > prevEvalEpoch and i % (int(N / lot_size) * lot_size) == 0):
        prevEvalEpoch = epoch

        self.coef_ = w / (self.lambda_ * N)

        cost = self.objective(X, y, self.coef_, alpha, self.lambda_)
        self.evaluate(X_test, Y_test, epoch, (time.time() - start_time)*1000, cost)
        if self.verbose:
          self._vecCompare(X.T.dot(alpha), w)

        numLots = 0

      n = perm[i % N]
      lot.append(n)

      delta_n = self._delta_alpha(y[n], w, X[n,:], alpha[n],
                                        self.lambda_, self.sigmaP, N)

      # per-example clipping
      if self.C > 0:
        scale = np.maximum(1, np.abs(delta_n) / self.C)
        delta_n /= scale

      # update dual vector and temp shared vector
      alpha[n] += delta_n
      if (isinstance(X, scipy.sparse.csr_matrix)):
        delta_w_temp = X[n,:].multiply(delta_n).transpose()
      else:
        delta_w_temp = X[n,:]*delta_n
      w += delta_w_temp

      # process one lot
      if (i+1) % lot_size == 0:
        # add noise
        a_noise = None
        w_noise = None
        if self.sigma > 0:
          alpha[lot] += np.random.normal(0, self.sigma * np.sqrt((4*lot_size**2-2)*self.C**2), size=len(lot))
          w += np.random.normal(0, self.sigma * np.sqrt((4*lot_size**2-2)*self.C**2), size=M)

        numLots += 1
        self.coef_ = w / (self.lambda_ * N) #primal model
        process_lot_i = i
        lot = [] # reset lot

    assert process_lot_i == i # output model is perturbed (if DP)

    cost = self.objective(X, y, self.coef_, alpha, self.lambda_)
    self.evaluate(X_test, Y_test, epoch+1, (time.time() - start_time)*1000,
                  cost, lastTrainingEpoch=True)

    # output useful for COCOA
    if not alpha_init is None and not w_init is None:
      self._delta_alpha_k = alpha - alpha_init
      self._delta_w = (w - w_init) / self.sigmaP

    return self


  def SGD(self, X, y, X_test, Y_test):
    """Stochastic Gradient Descent"""

    start_time = time.time()

    N, M = X.shape

    w = np.zeros(M)
    self.coef_ = w

    # derive lot_size scale lot_ratio to [0,1]
    if self.lot_ratio <= 1:
      lot_size = max(1, math.ceil(self.lot_ratio * N))
    else:
      lot_size = self.lot_ratio
      self.lot_ratio /= N

    # scale the total_updates such that the last update includes a lot processing
    total_updates = int(N * self.max_iter)
    if lot_size > total_updates:
      print("Truncating lot_size...")
      lot_size = total_updates
    self.lot_ratio = lot_size / N
    total_updates -= int(total_updates % lot_size)

    print("Lot_size: %d" % lot_size)
    print("Total_updates: %d" % total_updates)

    # list of privacy params for the output mechanism M = M_a + M_b + ...
    # each entry contains a (iterative) mechanism Mi (see accountant.py)
    # in this case: M = M_a = M_1 + M_2 + ... + M_k
    self.iters = [int(total_updates / lot_size)] # compositions for the updates
    self.q = [lot_size/N] # sampling probability for the updates
    print("Privacy params (q, iters): (%s, %s)" % (self.q, self.iters))
    self._privacy_accounting(eps=self.eps, q=self.q[0], iters=self.iters[0]) #, delta=1/N**2)

    perm = np.arange(N)

    epoch = -1
    total_grad = np.zeros(M)

    for i in range(total_updates):
      # completed one epoch
      if i % N == 0:
        epoch += 1

        np.random.shuffle(perm)

        cost = self.objective(X, y, w, None, self.lambda_)
        self.evaluate(X_test, Y_test, epoch, (time.time() - start_time)*1000, cost)

      # single example gradient
      n = perm[i % N]
      grad = self._gradient(y[n], w, X[n])
      # grad += 1/lot_size * self.lambda_ * w # TODO uncomment this and comment next todo

      # scale gradient
      if self.C > 0:
        grad /= np.maximum(1, np.linalg.norm(grad) / self.C)

      total_grad += grad

      # completed one lot
      if (i+1) % lot_size == 0:
        # TODO replacing the line below with the TODO above doesn't practically affect performance but is more correct
        total_grad += self.lambda_ * w

        # add noise
        if self.sigma > 0:
          if self.C > 0:
            scaled_sigma = self.sigma * self.C
          else:
            scaled_sigma = self.sigma
          total_grad += np.random.normal(0, scaled_sigma, size=M)

        total_grad /= lot_size
        w -= self.gamma * total_grad
        total_grad = np.zeros(M)
        self.coef_ = w

    cost = self.objective(X, y, w, None, self.lambda_)
    self.evaluate(X_test, Y_test, epoch+1, (time.time() - start_time)*1000,
                  cost, lastTrainingEpoch=True)

    return self

  def COCOA(self, X, y, X_test, Y_test):

    start_time = time.time()
    N, M = X.shape

    # partition data among self.K machines
    indices = np.random.choice(range(N), N, replace=False)
    partitions = np.array_split(indices, self.K)

    local_alphas = [np.zeros(len(partitions[k])) for k in range(self.K)]
    w = np.zeros(M)
    self.coef_ = w / self.lambda_

    # privacy params
    iters = []
    qs = []

    if self.verbose:
      print("For plotting run: for file in *; do awk '/eval:0/ {print $0} /COCOA epoch finished/ {getline; print $0}' $file > /path/to/temp/$file; done")
    # assume bk=1 (scaling parameter)
    verbose = self.verbose
    stdout = sys.stdout
    nullstdout = open(os.devnull, 'w')
    for t in range(self.out_iter):
      print("COCOA epoch finished")
      self.evaluate(X_test, Y_test, t, (time.time() - start_time)*1000)

      if self.verbose:
        # consistency check (w = Aa)
        alpha = np.zeros(N)
        for k in range(self.K):
          alpha[partitions[k]] = local_alphas[k]
        self._vecCompare(X.T.dot(alpha), w)

      delta_w_sum = np.zeros(M)
      self.verbose = False
      for k in range(self.K):
        if False and k > 0 or t > 0:
          sys.stdout = nullstdout
        #self.dual_PSCD(X[partitions[k]], y[partitions[k]], X_test, Y_test,
        self.dual_SSCD(X[partitions[k]], y[partitions[k]], X_test, Y_test,
            local_alphas[k], w)
        local_alphas[k] += self.gamma * self._delta_alpha_k

        delta_w_sum += self._delta_w

        iters.append(self.iters[0])
        qs.append(self.q[0] * len(partitions[k]) / N)

      self.verbose = verbose
      sys.stdout = stdout

      w += self.gamma * delta_w_sum
      self.coef_ = w / self.lambda_

    self.evaluate(X_test, Y_test, t+1, (time.time() - start_time)*1000)

    self.iters = iters
    self.q = qs

    return self

  def fit(self, X, y, X_test=None, Y_test=None):
    """Args:
      same as sklearn
      X_test, Y_test: validation set to monitor performance while training
    """
    if self.fit_intercept:
      if (isinstance(X, scipy.sparse.csr_matrix)):
        ones = scipy.sparse.csr_matrix(np.reshape(np.ones(X.shape[0]), [-1, 1]))
        X = scipy.sparse.hstack((ones, X), format='csr')
      else:
        X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)

    if self.solver == 'SSCD':
      if self.dual:
        return self.dual_SSCD(X, y, X_test, Y_test, alpha_init=None, w_init=None)
      else:
        return self.primal_CD(X, y, X_test, Y_test)
    elif self.solver == 'PSCD':
      return self.dual_PSCD(X, y, X_test, Y_test, alpha_init=None, w_init=None)
    elif self.solver == 'SGD':
      return self.SGD(X, y, X_test, Y_test)
    elif self.solver == 'COCOA':
      return self.COCOA(X, y, X_test, Y_test)
    else:
      raise(NotImplementedError("Unknown solver"))
