from greedyConstructionAgents import GreedyConstructionPiAgent
import numpy as np
import config
import lp

class FeatureBasedPolicyQueryAgent(GreedyConstructionPiAgent):
  def __init__(self, cmp, rewardSet, initialPhi, queryType, gamma):
    # step size for local search
    self.epsilon = 0.01
    # call the parent class.
    # FIXME note that query iteration needs to be turned on for this method, as discussed in our report
    GreedyConstructionPiAgent.__init__(self, cmp, rewardSet, initialPhi, queryType, gamma, qi=False)

  def findNextPolicy(self, S, A, R, T, s0, psi, maxV, q):
    """
    We exploit the fact that the reward candidates are linearly parameterizable.
    We find all the verticies formed by previous policies, and optimize EUS locally.
    
    Note that R is a set of parameters.
    """
    dimension = config.DIMENSION
    # first, we find all the verticies formed by occupancies in q and the borders [0, config.WEIGHT_MAX_VALUE] in each dimension
    # start by creating the matrix x^T \Phi \omega
    fullA = np.array((0, dimension))

    # add hyperplanes for policies in q
    for x in q:
      a = np.zeros((1, dimension))
      for i in range(dimension):
        # compute <\Phi_i, x> and add to this a
        a[i] = sum(self.cmp.getFeatures(s, act)[i] * x[s, act] for s in S for act in A)
      
      # add a vector of feature occupancy of policy x to fullA
      np.vstack((fullA, a))
    fullB = np.zeros((len(fullA), 1))
    
    # add borders
    for i in range(dimension):
      a = np.zeros(dimension)
      a[i] = 1
      # left border
      np.vstack((fullA, a))
      np.vstack((fullB, 0))
      # right border
      np.vstack((fullA, a))
      np.vstack((fullB, config.WEIGHT_MAX_VALUE))
      
    # now iterate over all verticies 
    maxEUS = -np.inf
    optPi = None
    from itertools import combinations
    for subset in combinations(range(len(fullA)), dimension + 1):
      # TODO filter the subset that have borders of the same dimension to make this process more efficient

      a = fullA[subset]
      b = fullB[subset]
      try:
        w = np.linalg.solve(a, b)
      except np.linalg.LinAlgError as err:
        if 'Singular matrix' in err.message:
          print 'this matrix is singular, continue'
          continue
        else:
          raise
      
      # optimize the reward parameter w and its neighbors to find the one with the highest EUS
      x = self.optimizeW(w)

      # compute EUS of union of q and {x}
      eus = lp.computeObj(q + [x], psi, S, A, R)
      if eus > maxEUS:
        maxEUS = eus
        optPi = x
    
    return optPi
