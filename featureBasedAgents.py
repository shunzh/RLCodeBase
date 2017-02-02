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
    print len(q)
    dimension = config.DIMENSION + 1 # V^\pi is the last variable
    # first, we find all the verticies formed by occupancies in q and the borders [0, config.WEIGHT_MAX_VALUE] in each dimension
    # start by creating the matrix x^T \Phi \omega
    fullA = np.zeros((0, dimension))

    # add hyperplanes for policies in q
    for x in q:
      a = np.zeros(dimension)
      for i in range(dimension - 1):
        # compute <\Phi_i, x> and add to this a
        a[i] = sum(self.cmp.getFeatures(s, act)[i] * x[s, act] for s in S for act in A)
      a[dimension - 1] = -1
      
      # add a vector of feature occupancy of policy x to fullA
      fullA = np.vstack((fullA, a))
    fullB = np.zeros((len(fullA), 1))
    
    # add borders
    for i in range(dimension - 1):
      a = np.zeros(dimension)
      a[i] = 1
      # left border
      fullA = np.vstack((fullA, a))
      fullB = np.vstack((fullB, -config.WEIGHT_MAX_VALUE))
      # right border
      fullA = np.vstack((fullA, a))
      fullB = np.vstack((fullB, config.WEIGHT_MAX_VALUE))
      
    # now iterate over all verticies 
    maxEUS = -np.inf
    optPi = None
    from itertools import combinations
    for subset in combinations(range(len(fullA)), dimension):
      # TODO filter the subset that have borders of the same dimension to make this process more efficient
      subset = list(subset)

      a = fullA[subset]
      b = fullB[subset]
      
      if np.linalg.matrix_rank(a) < dimension:
        print 'not solvable'
        continue

      try:
        w = np.linalg.solve(a, b)
      except np.linalg.LinAlgError as err:
        if 'Singular matrix' in err.message:
          print 'this matrix is singular, continue'
          continue
        else:
          raise
      
      print 'a', a
      print 'b', b
      print 'intersect at', w

      w = w[:-1] # the last one is the paramter for V

      # optimize the reward parameter w and its neighbors to find the one with the highest EUS
      if any(abs(wi) > config.WEIGHT_MAX_VALUE for wi in w): continue
      x = self.optimizeW(np.transpose(w))

      # compute EUS of union of q and {x}
      eus = lp.computeObj(q + [x], psi, S, A, R)
      if eus > maxEUS:
        maxEUS = eus
        optPi = x
    
    return optPi
