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
  def __init__(self, qFuncs, eta = 1, learnDiscounter = False, solver = "BFGS"):
    """
      Args:
        qFuncs: a list of Q functions for all the modules
        eta: consistency factor
        learnDiscounter: put discounter as part of X to solve.
                         Currently not doing in this way - would be nonconvex. 
    """
    self.qFuncs = qFuncs
    self.n = len(self.qFuncs)

    # enable if learning discounters as well
    self.learnDiscounter = learnDiscounter
    self.solver = solver

    # confidence
    self.eta = eta
    # default discounters
    self.d = [.8] * self.n
  
  def setDiscounters(self, d):
    # make sure the length of discounters is correct
    assert len(d) == self.n
    # set discounters here
    self.d = d

  def obj(self, X):
    """
      The objective function to be minimized.

      Args:
        X: parameter vector, weights and discounters.
      Return:
        - log(likelihood)
    """
    w = X[:self.n] # weights
    
    if self.learnDiscounter:
      self.d = X[self.n:]

    def computeQValue(state, action):
      # s, a -> q(s, a)
      return sum([w[moduleIdx] * self.qFuncs[moduleIdx](state, action, self.d) for moduleIdx in xrange(len(self.qFuncs))])
    
    # to be minimized, return the negation of log
    return - self.softMaxSum(computeQValue)

  def solve(self):
    """
      Overrides IRL's solver.
      Find the appropriate weight and discounter for each module, by walking through the policy.

      Return:
        optimal weight and discounter, in one vector
    """
    # make sure the range of weights are positive
    start_pos = [0] * self.n
    bnds = tuple((0, 1000) for _ in range(self.n))
    if self.learnDiscounter:
      # range of discounters
      margin = 0.1
      bnds += tuple((0 + margin, 1 - margin) for _ in range(self.n))
      start_pos += [0.5] * self.n

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

    sumX = sum(x[:self.n])
    try:
      w = [x[idx] / sumX for idx in xrange(self.n)]
    except:
      w = x[:self.n]
      print "all 0 weights. weird"
      
    d = x[self.n:]
    
    # concatenate weights and discounter (could be [])
    return w + d