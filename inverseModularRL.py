import numpy as np
from scipy.optimize import differential_evolution, minimize
from inverseRL import InverseRL
import config
import cma

class InverseModularRL(InverseRL):
  """
    Inverse Reinforcement Learning. From a trained modular agent, find the weights given
    - Q tables of modules
    - Policy function

    This is implemented as
    C. A. Rothkopf and Ballard, D. H.(2013), Modular inverse reinforcement
    learning for visuomotor behavior, Biological Cybernetics,
    107(4),477-490
    http://www.cs.utexas.edu/~dana/Biol_Cyber.pdf
  """
  def __init__(self, qFuncs, starts, bnds,\
               decorator = lambda x: x,\
               regular = lambda x: 0,\
               eta = 1, solver = "BFGS"):
    """
      Args:
        qFuncs: a list of Q functions for all the modules
        eta: consistency factor
        starts: init values of X
        bnds: boundary of X
    """
    self.qFuncs = qFuncs
    self.n = len(self.qFuncs)

    # initial values and bounds
    self.starts = starts
    self.bns = bnds
    # decorate X if necessary, identity function by default
    self.decorator = decorator 
    self.regular = regular

    self.solver = solver

    # confidence
    self.eta = eta
    # default discounters
  
  def obj(self, X):
    """
      The objective function to be minimized.

      Args:
        X: parameter vector, weights and discounters.
      Return:
        - log(likelihood)
    """
    # append the constants, which are fixed and not to solve
    reg = self.regular(X)
    X = self.decorator(X)

    def computeQValue(state, action):
      # s, a -> q(s, a)
      return sum([self.qFuncs[moduleIdx](state, action, X) for moduleIdx in xrange(len(self.qFuncs))])
    
    # to be minimized, return the negation of log
    return - self.softMaxSum(computeQValue) + reg

  def solve(self):
    """
      Return:
        optimal weight and discounter, in one vector
    """
    # make sure the range of weights are positive
    start_pos = self.starts
    bnds = self.bns

    if self.solver == "BFGS":
      # BFGS should be default for `minimize`
      result = minimize(self.obj, start_pos, bounds=bnds)
      x = result.x.tolist()
    elif self.solver == "DE":
      result = differential_evolution(self.obj, bnds)
      x = result.x.tolist()
    elif self.solver == "CMA-ES":
      xWrapper = lambda x: [bnds[idx][0] + x[idx] / 10.0 * (bnds[idx][1] - bnds[idx][0]) for idx in xrange(len(x))]
      cmaObj = lambda x: self.obj(xWrapper(x))
      result = cma.fmin(cmaObj, start_pos, 2,\
                        {'bounds': [0, 10]})
      x = xWrapper(result[0])
      print x
    else:
      raise Exception("Unknown solver " + self.solver)

    # concatenate weights and discounter (could be [])
    return x