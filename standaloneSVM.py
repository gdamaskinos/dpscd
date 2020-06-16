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
from standaloneLR import LogisticRegression


class SVM(LogisticRegression):
  """Standalone SVM with SCD"""

  def __init__(self, *args, box=0.0001, **kwargs):
    super(SVM, self).__init__(*args, **kwargs)
    self.box = box
    print("Standalone SVM")
    if self.dual:
      print("box: %f" % self.box)

  def _primal_loss(self, X, y, theta):
    """cocoa-update-derivations.pdf (Equation 34) - regularizer + scale loss by N"""
    N = y.shape[0]
    y_pred = X.dot(theta)

    sum_ = np.maximum(0, 1 - y_pred * y).sum()

    return 1/N * sum_

  def _dual_loss(self, X, y, alpha):
    """cocoa-update-derivations.pdf (Equation 35) - regularizer + scale loss by N"""
    N = X.shape[0]
    return 1/N * (-alpha.T.dot(y))

  # TODO
  def _delta_theta(self, yn, wn, Xn, theta_m, lambda_, sigmaP):
    pass

  def _delta_alpha(self, yn, w, Xn, alpha_n, lambda_, sigmaP, N):
    """lambda -> lambda*N in cocoa-update-derivations.pdf (Section 4.2)
    """
    alpha_n = self._enforce_alpha(yn, alpha_n, self.box)

    # power computation
    if scipy.sparse.issparse(Xn):
      temp = np.sum(Xn.power(2))
    else:
      temp = np.linalg.norm(Xn)**2

    delta = yn * max(0, min(1, yn * (
      alpha_n + (lambda_*N*yn - Xn.dot(w)) / (sigmaP * temp)))) - alpha_n

    # Compute new value of alpha (including constraint)
    new_alpha_n = self._enforce_alpha(yn, alpha_n + delta, self.box)

    # Compute new value of delta (after constraint is enforced)
    delta = new_alpha_n - alpha_n

    return delta

  def _gradient(self, yn, w, Xn):
    """Sub-gradient descent (reference:
      https://people.csail.mit.edu/dsontag/courses/ml16/slides/lecture5.pdf -> slide 16)"""
    ypred = Xn.dot(w)

    if yn * ypred < 1:
      return - yn * Xn
    else:
      return 0

