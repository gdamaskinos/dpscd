"""
  Copyright (c) 2020 Georgios Damaskinos
  All rights reserved.
  @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
  This source code is licensed under the MIT license found in the
  LICENSE file in the root directory of this source tree.
"""

import numpy as np
import pandas as pd

import time
import sys
import scipy
import pickle
from optimizer import Optimizer, print
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, log_loss, precision_recall_fscore_support


class LogisticRegression(Optimizer):
  """Standalone LR with SCD"""

  def __init__(self, *args, box=0.0001, **kwargs):
    super(LogisticRegression, self).__init__(*args, **kwargs)
    self.box = box
    print("Standalone LR")
    if self.dual:
      print("box: %f" % self.box)

  def _primal_loss(self, X, y, theta):
    """cocoa-update-derivations.pdf (Equation 23) - regularizer + scale loss by N"""
    N = y.shape[0]
    y_pred = X.dot(theta)

    sum_ = np.log(1 + np.exp(-y_pred * y)).sum()

    return 1/N * sum_

  def _dual_loss(self, X, y, alpha):
    """cocoa-update-derivations.pdf (First equation section 3.6) - regularizer + scale loss by N"""
    N = y.shape[0]
    sum_ = 0.0
    for i in range(N):
      alpha[i] = self._enforce_alpha(y[i], alpha[i], self.box)
      sum_ += (1-alpha[i]*y[i])*np.log(1-alpha[i]*y[i])
      sum_ += alpha[i]*y[i]*np.log(alpha[i]*y[i])

    return 1/N * sum_


  def _objective2(self, X, y_original, w, lambda_):
    """Based on logloss for y \in {0, 1}"""
    y = y_original.copy()
    N = y.shape[0]
    y[y == -1] = 0
    z = X.dot(w)
    h = 1 / (1 + np.exp(-z))
    sum_= (-y * np.log(h) - (1 - y) * np.log(1 - h)).sum()

    return 1/N * (sum_ + lambda_ * np.linalg.norm(w)**2)

  def _dual_optimal_dampening(self, y, alpha, w, delta_alpha, delta_w, sigma, lambda_):
    return -1

  def _primal_optimal_dampening(self, y, theta, w, delta_theta, delta_w, sigma, lambda_):
    pass

  def _enforce_alpha(self,y,alpha, box):
    clipped = np.clip(y * alpha, box, 1-box)
    return clipped * y

  def _compute_deltas2(self, y, w, delta_w, w_noise, alpha, delta_alpha, a_noise):
    """Box constraints by rescaling data
    NOT USED! instead add enforce_alpha in the beggining of the _delta_alpha
    and in the loop of the dual objective computation"""
    if not a_noise is None and not w_noise is None:
      delta_w += w_noise

      new_alpha = alpha + a_noise + delta_alpha
      clipped_alpha = self._enforce_alpha(y, new_alpha, self.box)
      delta_alpha = delta_alpha + (clipped_alpha - new_alpha)
      np.testing.assert_almost_equal(
        self._enforce_alpha(y, alpha + a_noise + delta_alpha, self.box),
        alpha + a_noise + delta_alpha)

      delta_alpha += a_noise

    return delta_w, delta_alpha

  def _delta_theta(self, yn, wn, Xn, theta_m, lambda_, sigmaP):
    # TODO
    pass

  def _delta_alpha(self, yn, w, Xn, alpha_n, lambda_, sigmaP, N):
    """lambda -> lambda*N and yi^2 = 1 in cocoa-update-derivations.pdf (page 10)
    Equivalent with dimitris_thesis.pdf (3.19) when yi <-> -yi (due to the
    difference in the dual loss definition) and yi^2 = 1
    """
    alpha_n = self._enforce_alpha(yn, alpha_n, self.box)

    coef_ = w / (lambda_ * N)
    der1 = -yn*np.log(1-alpha_n*yn) + yn*np.log(yn*alpha_n) + Xn.dot(coef_)
    # power computation
    if scipy.sparse.issparse(Xn):
      temp = np.sum(Xn.power(2))
    else:
      temp = np.linalg.norm(Xn)**2

    der2 = -1/(alpha_n*(alpha_n-yn)) + sigmaP/(lambda_ * N) * temp

    delta = -der1/der2
    # Compute new value of alpha (including constraint)
    new_alpha_n = self._enforce_alpha(yn, alpha_n + delta, self.box)

    # Compute new value of delta (after constraint is enforced)
    delta = new_alpha_n - alpha_n

    return delta

  def _gradient(self, yn, w, Xn):
    if yn == -1:
      yn = 0
    z = Xn.dot(w)
    h = 1 / (1 + np.exp(-z))
    error = h - yn
    return Xn.T.dot(error)

    # Significantly boosts the epoch=0 performance
#  def _init_alpha(self, y):
#    return -1/(2*y) # dual model

  # TODO port primal_CD to optimizer and merge with SPCA for ridge
  def primal_CD(self,X,y):
    """Primal Stochastic Coordinate Descent"""
    assert not scipy.sparse.issparse(X)
    start_time = time.time()
    N, M = X.shape
    alpha = np.zeros(M)
    w = np.zeros(N)
    perm = np.arange(M)
    if self.group_size == -1:
      self.group_size = N
    deltas = []
    for epoch in range(self.max_iter):
      primal_cost = self._primal_objective(X,y,alpha)
      if self.verbose:
        print("eval:%d,%d,%f,-1,-1,-1,%f" % (
          epoch, epoch, primal_cost, (time.time() - start_time)*1000))

      np.random.shuffle(perm)
      for j in range(M):
        m = perm[j]

        sample = np.random.choice(range(N), self.group_size, replace=False)
        der1 = -y[sample].dot(X[sample,m] * (np.exp(-y[sample]*w[sample]) / (
          1 + np.exp(-y[sample]*w[sample]))))
        der2 = np.dot(X[sample,m]**2, np.exp(-y[sample]*w[sample]) / (
          (1 + np.exp(-y[sample]*w[sample]))**2))

        der1 += self.lambda_*alpha[m]
        der2 += self.lambda_

        delta_alpha = -der1/der2

        # add noise
        if self.sigma > 0:
          delta_alpha += np.random.normal(0, self.sigma)

        deltas.append(float(delta_alpha))

        alpha[m] += delta_alpha

        w += X[:,m]*delta_alpha.flatten()

    primal_cost = self._primal_objective_opt(X,y,w,alpha)
    if self.verbose:
      print("eval:%d,%d,%f,-1,-1,-1,%f" % (
        epoch+1, epoch+1, primal_cost, (time.time() - start_time)*1000))

    with open("/home/gda/deltas_" + str(self.sigma) + ".pickle", 'wb') as handle:
      pickle.dump(deltas, handle)

    self.coef_ = alpha
    self.iters += M * self.max_iter
    return self


    self.coef_ = w / self.lambda_

    perm = np.arange(N)
    if self.verbose:
      print("round,epoch,trE,trA,valE,valA,Time(ms)")

    deltas = []
    for epoch in range(self.max_iter):
        dual_cost = self._dual_objective_opt(y,alpha,w,self.lambda_)
        if self.verbose:
          self.evaluate(X_test, Y_test, epoch, (time.time() - start_time)*1000, dual_cost)

        np.random.shuffle(perm)
        for i in range(N):
          n = perm[i]

          der1 = y[n]*(np.log(alpha[n]*y[n]+1)-np.log(-y[n]*alpha[n])) + (X[n].dot(w))/(self.lambda_)
          # power computation
          if scipy.sparse.issparse(X[n]):
            temp = X[n].power(2)
          else:
            temp = np.power(X[n], 2)

          der2 = -1/(alpha[n]*(alpha[n]+y[n])) + np.sum(temp)/self.lambda_

          delta = -der1/der2

          # add noise
          if self.sigma > 0:
            delta += np.random.normal(0, self.sigma, delta.shape)

          # Compute new value of alpha (including constraint)
          new_alpha_n = enforce_alpha(y[n], alpha[n] + delta, self.box)

          # Compute new value of delta (after constraint is enforced)
          delta = new_alpha_n - alpha[n]

          deltas.append(float(delta))

          # Update n-th coordinate
          alpha[n] += delta

          w += X[n].multiply(delta).transpose()

    if self.verbose:
      dual_cost = self._dual_objective_opt(y,alpha,w,self.lambda_)
      self.evaluate(X_test, Y_test, epoch+1, (time.time() - start_time)*1000, dual_cost)

    self.coef_ =  -w/self.lambda_
    self.iters = N * self.max_iter
    return self

  def predict(self, X):
    if self.fit_intercept:
      if (isinstance(X, scipy.sparse.csr_matrix)):
        ones = scipy.sparse.csr_matrix(np.reshape(np.ones(X.shape[0]), [-1, 1]))
        X = scipy.sparse.hstack((ones, X), format='csr')
      else:
        X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)

    y_pred = X.dot(self.coef_)
    threshold = 0
    y_pred[np.where(y_pred <= threshold)] = -1
    y_pred[np.where(y_pred >  threshold)] =  1

    return y_pred

  def evaluate(self, X_test, Y_test, epoch=-1, time=-1, cost=-1,
               lastTrainingEpoch=False):
    if np.isnan(cost) or np.isinf(cost):
      raise(ArithmeticError("Nan or inf Loss"))
    # test set evaluation
    if epoch == -1:
      y_pred = self.predict(X_test)
      print("Test accuracy: %g" % accuracy_score(Y_test, y_pred))
      prf = precision_recall_fscore_support(Y_test, y_pred, average='binary')
      print("perf: ", prf)
      tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()
      print('tp:',tp)
      print('fp:',fp)
      print('fn:',fn)
      print('tn:',tn)

    # training-validation set evaluation
    else:
      if epoch == 0:
        print("-1 indicates that the metric is not measured")
        print("round,epoch,training loss,training accuracy,validation loss,validation accuracy,Time(ms)")
      if not X_test is None:
        y_pred = self.predict(X_test)
        acc = accuracy_score(Y_test, y_pred)
        print("eval:%d,%d,%g,-1,-1,%g,%g" % (
                epoch, epoch, cost, acc, time))
        if lastTrainingEpoch:
          print("Validation accuracy: %g" % acc)
      else:
        print("eval:%d,%d,%g,-1,-1,-1,%g" % (
                epoch, epoch, cost, time))
