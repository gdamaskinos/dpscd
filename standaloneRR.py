"""
  Copyright (c) 2020 Georgios Damaskinos
  All rights reserved.
  @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
  This source code is licensed under the MIT license found in the
  LICENSE file in the root directory of this source tree.
"""

"""[paper]: https://arxiv.org/pdf/1702.07005.pdf
[SS13]: http://www.jmlr.org/papers/volume14/shalev-shwartz13a/shalev-shwartz13a.pdf
"""

import numpy as np
import pandas as pd

import time
import sys
import scipy
import sympy
import math
from sklearn.metrics import mean_squared_error, r2_score, pairwise
import sklearn
from optimizer import Optimizer, print

class RidgeRegression(Optimizer):
  """Standalone Ridge Regression with SCD"""

  def __init__(self, *args, **kwargs):
    super(RidgeRegression, self).__init__(*args, **kwargs)
    print("Standalone Ridge Regression")

  def _primal_loss(self, X, y, theta):
    """cocoa-update-derivations.pdf (Equation 10) - regularizer + scale loss by N"""
    N = X.shape[0]
    loss = 1/N * 1/2 * np.linalg.norm(X.dot(theta) - y) ** 2
    return loss

  def _dual_loss(self, X, y, alpha):
    """cocoa-update-derivations.pdf (after Equation 10) - regularizer + scale loss by N"""
    N = X.shape[0]
    return 1/N * (1/2 * np.linalg.norm(alpha) ** 2 - alpha.T.dot(y))


  def _dual_optimal_dampening2(self, y, alpha, w, delta_alpha, delta_w, sigma, lambda_):
    """Return argmin_gamma{dual_objective_opt(
      alpha+gamma*(delta_alpha + sigma * ||delta_alpha|| * za),
      w + gamma*(delta_w + sigma * ||delta_w|| * zw))}
    """
    N = alpha.shape[0]
    M = w.shape[0]

    z = np.random.normal(0, sigma)
    za = np.random.normal(0, sigma * np.linalg.norm(delta_alpha), size=N)
    zw = np.random.normal(0, sigma * np.linalg.norm(delta_w), size=M)
    #za = np.linalg.norm(delta_alpha) * z
    #zw = np.linalg.norm(delta_w) * z

    ca = delta_alpha + za
    cw = delta_w + zw

    # derivatives of the 3 terms in self.dual_objective for the custom v1, v2
    #gamma = sympy.symbols('gamma')
    #term1 = np.sum((alpha + gamma * ca) * ca
    #term2 = 1/lambda_ * np.sum((w + gamma * cw) * cw
    #term3 = -ca.T.dot(y)
    #gamma_min = float(sympy.solve([term1 + term2 + term3], gamma)[gamma])

    # formula almost same as the last equation in page 7 (https://arxiv.org/pdf/1702.07005.pdf)
    # the difference is a) due to modified objective (see delta computation)
    # b) the equation in the paper has a typo: \alpha -> \Delta\alpha (denominator 2nd term)
    gamma_min = (ca.T.dot(y) - ca.T.dot(alpha) - 1/lambda_ * cw.T.dot(w)) / (
        1/lambda_ * np.linalg.norm(cw) ** 2 + np.linalg.norm(ca)**2)

    #print("Pseudo objective: %g" % self.dual_objective_opt(y, alpha+gamma_min*ca, w+gamma_min*cw, lambda_))
    #for test_gamma in [1.001 * gamma_min, 0.999 * gamma_min]:
    #  print("test: ", self.dual_objective_opt(y, alpha+test_gamma*ca, w+test_gamma*cw, lambda_))

    return gamma_min

  def _primal_optimal_dampening(self, y, theta, w, delta_theta, delta_w, sigma, lambda_):
    """Return argmin_gamma{E[primal_objective_opt(
      theta+gamma*(delta_theta + sigma * ||delta_theta|| * zb),
      w + gamma*(delta_w + sigma * ||delta_w|| * zw))]}
    """

    N = w.shape[0]
    M = theta.shape[0]

    delta_theta_norm = np.linalg.norm(delta_theta)
    delta_w_norm = np.linalg.norm(delta_w)

    cb = sigma**2 * np.linalg.norm(delta_theta)**2 * M
    cw = sigma**2 * np.linalg.norm(delta_w)**2 * N

    gamma_min = -((w-y).dot(delta_w) + lambda_ * theta.dot(delta_theta)) / (
      delta_w_norm ** 2 + lambda_ * delta_theta_norm ** 2 + lambda_ * cb + cw)

    #print("Pseudo objective: %g" % self._primal_objective_opt(y, theta+gamma_min*cb, w+gamma_min*cw, lambda_))
    #for test_gamma in [1.001 * gamma_min, 0.999 * gamma_min]:
    #  print("test: ", self._primal_objective_opt(y, theta+test_gamma*cb, w+test_gamma*cw, lambda_))

    return gamma_min

  def _dual_optimal_dampening(self, y, alpha, w, delta_alpha, delta_w, sigma, lambda_):
    """Return argmin_gamma{E[dual_objective_opt(
      alpha+gamma*(delta_alpha + sigma * ||delta_alpha|| * za),
      w + gamma*(delta_w + sigma * ||delta_w|| * zw))]}
    """

    N = alpha.shape[0]
    M = w.shape[0]

    delta_alpha_norm = np.linalg.norm(delta_alpha)
    delta_w_norm = np.linalg.norm(delta_w)

    ca = sigma**2 * np.linalg.norm(delta_alpha)**2 * N
    cw = sigma**2 * np.linalg.norm(delta_w)**2 * M

    gamma_min = (delta_alpha.T.dot(y) - delta_alpha.T.dot(alpha) - 1/lambda_ * delta_w.T.dot(w)) / (
      1/lambda_ * delta_w_norm ** 2 + delta_alpha_norm**2 + 1/lambda_ * cw + ca)

    #print("Pseudo objective: %g" % self.dual_objective_opt(y, alpha+gamma_min*ca, w+gamma_min*cw, lambda_))
    #for test_gamma in [1.001 * gamma_min, 0.999 * gamma_min]:
    #  print("test: ", self.dual_objective_opt(y, alpha+test_gamma*ca, w+test_gamma*cw, lambda_))

    return gamma_min


  def _delta_theta(self, yn, wn, Xn, theta_m, lambda_, sigmaP, N):
    # w = X.dot(theta) # iff sample_ratio=1 and sigma=0
    """self.lambda = lambda * N [paper]"""
    #return ((yn - wn).dot(Xn) - self.lambda_ * theta_m) / (
    #  np.linalg.norm(Xn)**2 + self.lambda_)

    """follows [paper]"""
    return ((yn - wn).dot(Xn) - self.lambda_ * N * theta_m) / (
      np.linalg.norm(Xn)**2 + self.lambda_ * N)

  def _delta_alpha(self, yn, w, Xn, alpha_n, lambda_, sigmaP, N):
    """lambda = lambda * N [paper]
      self.lambda_ * y = lambda * y [paper] due to the difference in the objective function
    """
    #return (lambda_ * yn - w.T.dot(Xn) - lambda_*N*alpha_n) / (
    #          sigmaP * np.linalg.norm(Xn)**2 + lambda_ * N)

    """lambda = lambda * N in [cocoa-update-derivations.pdf](Section 2.4)
    Equivalent with [SS13](Section 6.2 with loss 1/2 ||...||^2 => 0.5 -> 1)
    !! the difference with the [paper]-based update rul is only in the DP setting
    """
    coef_ = w / (lambda_ * N)
    return (yn - coef_.T.dot(Xn) - 1 * alpha_n) / (
              1 + sigmaP * np.linalg.norm(Xn)**2 / (lambda_ * N))

  def _gradient(self, yn, w, Xn):
    error = Xn.dot(w) - yn
    return Xn.T.dot(error)

  def predict(self, X_test):
    if self.fit_intercept:
      X_test = np.insert(X_test, 0, values=np.ones(X_test.shape[0]), axis=1)
#    y_pred = np.dot(X_test, self.coef_)
    y_pred = X_test.dot(self.coef_)

    return y_pred

  def evaluate(self, X_test, Y_test, epoch=-1, time=-1, cost=-1,
               lastTrainingEpoch=False):
    if np.isnan(cost) or np.isinf(cost):
      raise(ArithmeticError("Nan or inf Loss"))
    # test set evaluation
    if epoch == -1:
      y_pred = self.predict(X_test)
      print("Test MSE: %g" % mean_squared_error(Y_test, y_pred))
      print("Test R2: %g" % r2_score(Y_test, y_pred))
    # training-validation set evaluation
    else:
      if epoch == 0:
        print("-1 indicates that the metric is not measured")
        print("round,epoch,training loss,training mse,validation loss,validation mse,Time(ms)")
      if not X_test is None:
        y_pred = self.predict(X_test)
        MSE = mean_squared_error(Y_test, y_pred)
        R2 = r2_score(Y_test, y_pred)
        if epoch == 0:
          self.MSE = MSE
        elif MSE > self.MSE * 3:  # early stopping
          raise(ArithmeticError("MSE > init MSE * 3\nTraining complete!"))
        print("eval:%d,%d,%f,-1,-1,%f,%f" % (
                epoch, epoch, cost, MSE, time))
        if lastTrainingEpoch:
          print("Validation MSE: %g" % MSE)
          print("Validation R2: %g" % R2)
      else:
        print("eval:%d,%d,%f,-1,-1,-1,%f" % (
                epoch, epoch, cost, time))
