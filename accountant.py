"""
  Copyright (c) 2020 Georgios Damaskinos
  All rights reserved.
  @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
  This source code is licensed under the MIT license found in the
  LICENSE file in the root directory of this source tree.
"""

"""Based on https://github.com/rwightman/tensorflow-models/blob/master/research/differential_privacy/privacy_accountant/tf/accountant.py
"""


import abc
import collections
import math
import sys

import numpy as np
from numpy.random import normal, uniform
from scipy.stats import norm, binom


EpsDelta = collections.namedtuple("EpsDelta", ["spent_eps", "spent_delta", "min_lambda"])


def GenerateBinomialTable(m):
  """Generate binomial table.

  Args:
    m: the size of the table.
  Returns:
    A two dimensional array T where T[i][j] = (i choose j),
    for 0<= i, j <=m.
  """

  table = np.zeros((m + 1, m + 1), dtype=np.float64)
  for i in range(m + 1):
    table[i, 0] = 1
  for i in range(1, m + 1):
    for j in range(1, m + 1):
      v = table[i - 1, j] + table[i - 1, j -1]
      assert not math.isnan(v) and not math.isinf(v)
      table[i, j] = v
  return table

class MomentsAccountant(object):
  """Privacy accountant which keeps track of moments of privacy loss.

  Note: The constructor of this class creates tf.Variables that must
  be initialized with tf.global_variables_initializer() or similar calls.

  MomentsAccountant accumulates the high moments of the privacy loss. It
  requires a method for computing differenital moments of the noise (See
  below for the definition). So every specific accountant should subclass
  this class by implementing _differential_moments method.

  Denote by X_i the random variable of privacy loss at the i-th step.
  Consider two databases D, D' which differ by one item. X_i takes value
  log Pr[M(D')==x]/Pr[M(D)==x] with probability Pr[M(D)==x].
  In MomentsAccountant, we keep track of y_i(L) = log E[exp(L X_i)] for some
  large enough L. To compute the final privacy spending,  we apply Chernoff
  bound (assuming the random noise added at each step is independent) to
  bound the total privacy loss Z = sum X_i as follows:
    Pr[Z > e] = Pr[exp(L Z) > exp(L e)]
              < E[exp(L Z)] / exp(L e)
              = Prod_i E[exp(L X_i)] / exp(L e)
              = exp(sum_i log E[exp(L X_i)]) / exp(L e)
              = exp(sum_i y_i(L) - L e)
  Hence the mechanism is (e, d)-differentially private for
    d =  exp(sum_i y_i(L) - L e).
  We require d < 1, i.e. e > sum_i y_i(L) / L. We maintain y_i(L) for several
  L to compute the best d for any give e (normally should be the lowest L
  such that 2 * sum_i y_i(L) / L < e.

  We further assume that at each step, the mechanism operates on a random
  sample with sampling probability q = batch_size / total_examples. Then
    E[exp(L X)] = E[(Pr[M(D)==x / Pr[M(D')==x])^L]
  By distinguishing two cases of whether D < D' or D' < D, we have
  that
    E[exp(L X)] <= max (I1, I2)
  where
    I1 = (1-q) E ((1-q) + q P(X+1) / P(X))^L + q E ((1-q) + q P(X) / P(X-1))^L
    I2 = E (P(X) / ((1-q) + q P(X+1)))^L

  In order to compute I1 and I2, one can consider to
    1. use an asymptotic bound, which recovers the advance composition theorem;
    2. use the closed formula (like GaussianMomentsAccountant);
    3. use numerical integration or random sample estimation.

  Dependent on the distribution, we can often obtain a tigher estimation on
  the moments and hence a more accurate estimation of the privacy loss than
  obtained using generic composition theorems.

  """

  __metaclass__ = abc.ABCMeta

  def __init__(self, moment_orders=32, verbose=True, seed=1):
    """Initialize a MomentsAccountant.

    Args:
      moment_orders: the order of moments to keep.
    """

    np.random.seed(seed)

    self._moment_orders = (moment_orders
                           if isinstance(moment_orders, (list, tuple))
                           else range(1, moment_orders + 1))
    self._max_moment_order = max(self._moment_orders)
    self._log_moments = np.zeros(len(self._moment_orders))
    self._verbose = verbose

  def compute_sigma2(self, eps, delta, q, iters):
    """Compute sigma by doing a tenary+binary search on the moments accountant
    Faster and more accurate than compute_sigma()"""
    def magnitude(x):
      return int(math.log10(x))
    low = 1e-10
    high = 1e10

    while low <= high:
      mag1 = magnitude(low)
      mag2 = magnitude(high)
      if (mag2-mag1 > 1): # tenary search
        mid = 10**(int((mag1+mag2)/2))
      else:
        mid = (low+high)/2
      self.accumulate_privacy_spending(sigma=mid, q=q, iters=iters, reset=True)
      mid_eps = self.get_privacy_spent(target_deltas=[delta])[0][0]

      if eps == mid_eps:
        low = mid
        break
      elif eps > mid_eps:
        high = mid*0.99
      else:
        low = mid*1.01

    return low

  def compute_sigma(self, eps, delta, q, iters):
    """Compute sigma by doing a line search on the moments accountant"""
    sigma = 0.0001
    while True:
      self.accumulate_privacy_spending(sigma=sigma, q=q, iters=iters,
                                       reset=True)
      spent_delta = self.get_privacy_spent(target_eps=[eps])[0][1]
      if spent_delta <= delta:
        return sigma

      sigma *= 1.1


  @abc.abstractmethod
  def _compute_log_moment(self, sigma, q, moment_order):
    """Compute high moment of privacy loss.

    Args:
      sigma: the noise sigma, in the multiples of the sensitivity.
      q: the sampling ratio.
      moment_order: the order of moment.
    Returns:
      log E[exp(moment_order * X)]
    """
    pass

  def accumulate_privacy_spending(self, sigma, q, iters=1, reset=False):
    """Accumulate privacy spending.

    In particular, accounts for privacy spending when we assume there
    are num_examples, and we are releasing the vector
    (sum_{i=1}^{num_examples} x_i) + Normal(0, stddev=l2norm_bound*sigma)
    where l2norm_bound is the maximum l2_norm of each example x_i, and
    the num_examples have been randomly selected out of a pool of total_examples.

    Args:
      sigma: the noise sigma, in the multiples of the sensitivity (that is,
        if the l2norm sensitivity is k, then the caller must have added
        Gaussian noise with stddev=k*sigma to the result of the query).
      q: sampling probability (batch_size / num_examples).
      iters: number of times that noise is added (e.g., #epochs)
      reset: True -> resets accumulator
    """

    if reset:
      self._log_moments = np.zeros(len(self._moment_orders))

    # the following are useful for computing privacy if all moments are nan/inf
    self.q = q
    self.sigma = sigma
    self.iters = iters

    for i in range(len(self._log_moments)):
      alpha_i = self._compute_log_moment(sigma, q, self._moment_orders[i])

      # composability (Theorem 2.1)
      alpha = iters * alpha_i

      self._log_moments[i] += alpha


  def _compute_delta(self, log_moments, eps):
    """Compute delta for given log_moments and eps.

    Args:
      log_moments: the log moments of privacy loss, in the form of pairs
        of (moment_order, log_moment)
      eps: the target epsilon.
    Returns:
      delta
      min_lambda: moment that gives the minimum value
    """
    min_delta = 1.0
    min_lambda = None
    nanInfMoment = []
    valid = False
    for moment_order, log_moment in log_moments:
      if math.isinf(log_moment) or math.isnan(log_moment):
        nanInfMoment.append(moment_order)
        continue
      valid = True
      if log_moment < moment_order * eps:
        temp = math.exp(log_moment - moment_order * eps)
        if min_delta > temp:
          min_delta = temp
          min_lambda = moment_order

    if self._verbose:
      print("Inf or Nan moment orders: %s\n" % nanInfMoment)
    if not valid:
      # avoid numerical instability (inf) and directly compute delta
      # from formula to compute E2 (GaussianMomentsAccountant2) by setting k=1
      if self._verbose:
        print("All moments are inf or Nan")
      if self._verbose:
        print("Estimating privacy given min_lambda=1 from last accumulated sigma")
      min_delta = np.exp(self.iters * (np.log(self.q) + 1.0 / self.sigma**2) - eps)

    return min_delta, min_lambda

  def _compute_eps(self, log_moments, delta):
    min_eps = float("inf")
    min_lambda = None
    self._eps = []
    nanInfMoment = []
    valid = False
    for moment_order, log_moment in log_moments:
      if math.isinf(log_moment) or math.isnan(log_moment):
        nanInfMoment.append(moment_order)
        self._eps.append(None)
        continue
      valid=True
      temp = (log_moment - math.log(delta)) / moment_order
      self._eps.append(temp)

      if min_eps > temp:
        min_eps = temp
        min_lambda = moment_order
    if self._verbose:
      print("Inf or Nan moment orders: %s\n" % nanInfMoment)
    if not valid:
      # avoid numerical instability (inf) and directly compute delta
      # from formula to compute E2 (GaussianMomentsAccountant2) by setting k=1
      if self._verbose:
        print("All moments are inf or Nan")
      if self._verbose:
        print("Estimating privacy min_lambda=1 from last accumulated sigma")
      min_eps = self.iters * (np.log(self.q) + 1.0 / self.sigma**2) - np.log(delta)

    return min_eps, min_lambda

  def get_privacy_spent(self, target_eps=None, target_deltas=None):
    """Compute privacy spending in (e, d)-DP form for a single or list of eps.

    Args:
      target_eps: a list of target epsilon's for which we would like to
        compute corresponding delta value.
      target_deltas: a list of target deltas for which we would like to
        compute the corresponding eps value. Caller must specify
        either target_eps or target_delta.
    Returns:
      A list of EpsDelta pairs.
    """
    assert (target_eps is None) ^ (target_deltas is None)
    eps_deltas = []
    log_moments_with_order = zip(self._moment_orders, self._log_moments)
    if target_eps is not None:
      for eps in target_eps:
        delta, min_lambda = self._compute_delta(log_moments_with_order, eps)
        eps_deltas.append(EpsDelta(eps, delta, min_lambda))
    else:
      assert target_deltas
      for delta in target_deltas:
        eps, min_lambda = self._compute_eps(log_moments_with_order, delta)
        eps_deltas.append(EpsDelta(eps, delta, min_lambda))
    return eps_deltas


class GaussianMomentsAccountant(MomentsAccountant):
  """MomentsAccountant which assumes Gaussian noise.

  GaussianMomentsAccountant assumes the noise added is centered Gaussian
  noise N(0, sigma^2 I). In this case, we can compute the differential moments
  accurately using a formula.

  For asymptotic bound, for Gaussian noise with variance sigma^2, we can show
  for L < sigma^2,  q L < sigma,
    log E[exp(L X)] = O(q^2 L^2 / sigma^2).
  Using this we derive that for training T epoches, with batch ratio q,
  the Gaussian mechanism with variance sigma^2 (with q < 1/sigma) is (e, d)
  private for d = exp(T/q q^2 L^2 / sigma^2 - L e). Setting L = sigma^2,
  Tq = e/2, the mechanism is (e, exp(-e sigma^2/2))-DP. Equivalently, the
  mechanism is (e, d)-DP if sigma = sqrt{2 log(1/d)}/e, q < 1/sigma,
  and T < e/(2q). This bound is better than the bound obtained using general
  composition theorems, by an Omega(sqrt{log k}) factor on epsilon, if we run
  k steps. Since we use direct estimate, the obtained privacy bound has tight
  constant.

  I1 -> E2 (Equation 4)
  I2 -> E1 (Equation 3)
  For GaussianMomentAccountant, it suffices to compute I1, as I1 >= I2,
  which reduce to computing E(P(x+s)/P(x+s-1) - 1)^i for s = 0 and 1. In the
  companion gaussian_moments.py file, we supply procedure for computing both
  I1 and I2 (the computation of I2 is through multi-precision integration
  package). It can be verified that indeed I1 >= I2 for wide range of parameters
  we have tried, though at the moment we are unable to prove this claim.

  We recommend that when using this accountant, users independently verify
  using gaussian_moments.py that for their parameters, I1 is indeed larger
  than I2. This can be done by following the instructions in
  gaussian_moments.py.
  """

  def __init__(self, moment_orders=32, verbose=True, seed=1):
    """Initialization.

    Args:
      moment_orders: the order of moments to keep.
    """
    super(self.__class__, self).__init__(moment_orders, verbose, seed=seed)
    self._binomial_table = GenerateBinomialTable(self._max_moment_order)

  def _differential_moments(self, sigma, s, t):
    """Compute 0 to t-th differential moments for Gaussian variable.

        E[(P(x+s)/P(x+s-1)-1)^t]
      = sum_{i=0}^t (t choose i) (-1)^{t-i} E[(P(x+s)/P(x+s-1))^i]
      = sum_{i=0}^t (t choose i) (-1)^{t-i} E[exp(-i*(2*x+2*s-1)/(2*sigma^2))]
      = sum_{i=0}^t (t choose i) (-1)^{t-i} exp(i(i+1-2*s)/(2 sigma^2))
    Args:
      sigma: the noise sigma, in the multiples of the sensitivity.
      s: the shift.
      t: 0 to t-th moment.
    Returns:
      0 to t-th moment as an array of shape [t+1].
    """
    assert t <= self._max_moment_order, ("The order of %d is out "
                                         "of the upper bound %d."
                                         % (t, self._max_moment_order))
    binomial = self._binomial_table[0:t+1, 0:t+1]
    signs = np.zeros((t + 1, t + 1), dtype=np.float64)
    for i in range(t + 1):
      for j in range(t + 1):
        signs[i, j] = 1.0 - 2 * ((i - j) % 2)
    exponents = [j * (j + 1.0 - 2.0 * s) / (2.0 * sigma * sigma)
                             for j in range(t + 1)]
    # x[i, j] = binomial[i, j] * signs[i, j] = (i choose j) * (-1)^{i-j}
    x = binomial * signs
    # y[i, j] = x[i, j] * exp(exponents[j])
    #         = (i choose j) * (-1)^{i-j} * exp(j(j-1)/(2 sigma^2))
    # Note: this computation is done by broadcasting pointwise multiplication
    # between [t+1, t+1] tensor and [t+1] tensor.
    np.seterr(over='ignore', invalid='ignore')
    y = x * np.exp(exponents)
    # z[i] = sum_j y[i, j]
    #      = sum_j (i choose j) * (-1)^{i-j} * exp(j(j-1)/(2 sigma^2))
    z = np.sum(y, 1)
    return z

  def _compute_log_moment(self, sigma, q, moment_order):
    """Compute high moment of privacy loss.

    Args:
      sigma: the noise sigma, in the multiples of the sensitivity.
      q: the sampling ratio.
      moment_order: the order of moment.
    Returns:
      log E[exp(moment_order * X)]
    """
    assert moment_order <= self._max_moment_order, ("The order of %d is out "
                                                    "of the upper bound %d."
                                                    % (moment_order,
                                                       self._max_moment_order))

    # http://www.wolframalpha.com/input/?i=Solve%5Be%5E(t(t%2B1)%2F(2*sigma%5E2))+%3C+1.7976931348623157e%2B308,+sigma+%3E+0,+t+%3E+0,+sigma%5D
#    min_sigma = 0.0265413 * np.sqrt(moment_order*(moment_order+1))
#    assert sigma > min_sigma, (
#        "sigma < %f => inf value for the exponential calculations" % min_sigma)

    binomial_table = self._binomial_table[moment_order:moment_order+1, 0:moment_order+1]
    # qs = [1 q q^2 ... q^L] = exp([0 1 2 ... L] * log(q))
    qs = np.exp(np.array([range(moment_order + 1)]) * np.log(q))
    moments0 = self._differential_moments(sigma, 0.0, moment_order)
    term0 = np.sum(binomial_table * qs * moments0)
    moments1 = self._differential_moments(sigma, 1.0, moment_order)
    term1 = np.sum(binomial_table * qs * moments1)
    I1 = np.squeeze(q * term0 + (1.0 - q) * term1)

    try:
      self._E2s.append(I1) # I1 -> E2
    except AttributeError:
      self._E2s = []
      self._E2s.append(I1)

    return np.log(I1)

class GaussianMomentsAccountant2(MomentsAccountant):
  """Closed form computation - equivalent to GaussianMomentsAccountant (assume E2 > E1) b
  but with different formulas (see moment_accountant_E2.pdf)
  and much slower (~5x) that GaussianMomentsAccountant
  where
    E1 = E_{z~mu0}[mu_0(z)/mu(z)]
    E2 = E_{z~mu}[mu(z)/mu_0(z)]
    and
    mu(z) = (1-q)mu_0(z) + q mu_1(z)
    mu_0(z) ~ N(0,sigma^2)
    mu_1(z) ~ N(1,sigma^2)
  """
  def __init__(self, moment_orders=32, verbose=True, seed=1):
    """Initialization.

    Args:
      moment_orders: the order of moments to keep.
    """
    super(self.__class__, self).__init__(moment_orders, verbose, seed=seed)

  def _compute_log_moment(self, sigma, q, lam):
    """Compute high moment of privacy loss.

    Args:
      sigma: the noise sigma, in the multiples of the sensitivity.
      q: the sampling ratio.
      lam: the order of moment.
    Returns:
      alpha_lam = log E[exp(lam * X)] = log max(E1,E2) = log E2
        (assumption: E2 > E1)
    """
    assert lam <= self._max_moment_order, ("The order of %d is out "
                                                    "of the upper bound %d."
                                                    % (lam,
                                                       self._max_moment_order))
    E2 = 0
    for k in range(0, lam+1):
      E2 += binom.pmf(k, lam, q) * ((1-q)*np.exp(k*(k-1)/(2.0*sigma**2)) +
          q*np.exp(k*(k+1)/(2.0*sigma**2)))

    try:
      self._E2s.append(E2)
    except AttributeError:
      self._E2s = []
      self._E2s.append(E2)

    return np.log(E2)


class SamplingGaussianMomentsAccountant(MomentsAccountant):
  """Numerically approximates E1, E2 with random sampling from the distributions"""

  def __init__(self, moment_orders=32, verbose=False, seed=1):
    """Initialization.

    Args:
      moment_orders: the order of moments to keep.
    """
    super(self.__class__, self).__init__(moment_orders, verbose, seed=seed)

  def accumulate_privacy_spending(self, sigma, q, iters=1, N=None):
    """
    Args:
      N (int): number os samples to use for computing expectations
    """

    # section 3.2 (Equations 3,4)
    # check beginning of proof on Appendix B for explanation

    if N is None:
      # ensure that the bad coordinate is picked ~100 times
      N = int(100 / q)

    # generate sample from mean-0 Gaussian
    rng0 = normal(0, sigma, N)

    # generate sample from mean-1 Gaussian
    rng1 = normal(1, sigma, N)

    # generate sample from uniform distribution (needed for mixture)
    unif = uniform(size=N)

    # generate sample from mixture distribution
    rng = [ rng1[i] if unif[i] < q else rng0[i] for i in range(0,N) ]

    # Equation 3
    self._d1 = norm.pdf(rng0,0,sigma) / ((1-q)*norm.pdf(rng0, 0, sigma) + q*norm.pdf(rng0,1,sigma))
    # Equation 4
    self._d2 = ((1-q)*norm.pdf(rng, 0, sigma) + q*norm.pdf(rng, 1, sigma)) / norm.pdf(rng,0,sigma)

    # accumulate Equation 3,4 values (usefull for visualizing)
    self._E1s = []
    self._E2s = []

    super(self.__class__, self).accumulate_privacy_spending(
        sigma, q, iters)

  def _compute_log_moment(self, sigma, q, moment_order):
    E1 = np.mean(self._d1 ** moment_order)
    E2 = np.mean(self._d2 ** moment_order)

    self._E1s.append(E1)
    self._E2s.append(E2)

    alpha = np.log(max(E1, E2))
    return alpha

class NumericIntegrGaussianMomentsAccountant(MomentsAccountant):
  """Estimates E1, E2 with numerical integration"""

  def __init__(self, moment_orders=32, verbose=False, seed=1):
    """Initialization.

    Args:
      moment_orders: the order of moments to keep.
    """
    super(self.__class__, self).__init__(moment_orders, verbose, seed=seed)

  def accumulate_privacy_spending(self, sigma, q, iters=1, range_=30):
    """
    Args:
      range_ (float): numeric integral
    """
    z = np.linspace(-range_*sigma, 1.0+range_*sigma, 1000000)
    self._mu0 = norm.pdf(z, 0.0, sigma)
    self._mu1 = norm.pdf(z, 1.0, sigma)
    self._mu  = (1-q)*self._mu0 + q*self._mu1
    self._step = z[1]-z[0]

    self._E1s = []
    self._E2s = []

    super(self.__class__, self).accumulate_privacy_spending(
        sigma, q, iters)

  def _compute_log_moment(self, sigma, q, lam):
    E1 = np.sum(self._mu0 * (self._mu0 / self._mu) ** lam) * self._step
    E2 = np.sum(self._mu * (self._mu/self._mu0) ** lam) * self._step

    assert(E1 < E2) # check assumption of GaussianMomentAccountant
    self._E1s.append(E1)
    self._E2s.append(E2)

    alpha = np.log(max(E1, E2))
    return alpha


class DummyAccountant(object):
  """An accountant that does no accounting."""

  def accumulate_privacy_spending(self, *unused_args):
    return

  def get_privacy_spent(self, **unused_kwargs):
    return [EpsDelta(np.inf, 1.0, 1)]


def epsilonSigma(out, sigmaRange, q, iters=1, delta=0.00001):
  """Get epsilon = f(sigma) given delta and save it to out csv file"""

  eps = []
  sigmas = []
  for sigma in sigmaRange:
    acc = GaussianMomentsAccountant()
    acc.accumulate_privacy_spending(None, sigma, q, iters)
    eps.append(acc.get_privacy_spent(target_deltas=[delta])[0][0])
    sigmas.append(sigma)
    print((sigmas[-1], eps[-1]))

  np.savetxt(out, np.stack((sigmas, eps), axis=1), delimiter=',')
