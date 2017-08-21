from lp import lpDual, domPiMilp
import pprint

class ConsQueryAgent():
  """
  Find queries in constraint-uncertain mdps. May formulate constraints as negative rewards.

  TODO only implementing some auxiliary functions. 
  """
  def __init__(self, mdp, consIdx):
    """
    can't think of a class it should inherit..

    mdp: a factored mdp
    consIdx: specify a set of indices of features that the robot is not supposed to change without querying 
    """
    self.mdp = mdp
    self.consIdx = consIdx
  
  def findIrrelevantFeats(self):
    """
    DEPRECIATED. the crietrion is too strong. unlikely to rule out any features.
    use the dual lp problem to solve such problem.

    return: (occupancy measure, optimal value)
    """
    args = self.mdp
    featLength = len(args['S'][0])
    s0 = args['s0']
    A = args['A']
    
    # solve the raw problem
    constraints = {(s, a): 0 for a in A
                             for idx in self.consIdx
                             for s in self.statesWithDifferentFeats(idx, s0[idx])}
    args['constraints'] = constraints
    args['positiveConstraints'] = []
    rawOpt, occ = lpDual(**args)
    print rawOpt
    for s, prob in occ.items():
      if prob > 0: print s, prob

    irrFeats = []
    # solve the problem which only constrains one feature
    for idx in self.consIdx:
      print idx
      constraints = self.statesTransitToDifferentFeatures(idx, s0[idx])
      args['constraints'] = {}
      args['positiveConstraints'] = constraints
      opt, occ = lpDual(**args)
      print opt
      for s, prob in occ.items():
        if prob > 0: print s, prob
      
      if opt <= rawOpt: irrFeats.append(idx)
    
    return irrFeats
  
  def findDominatingPolicies(self):
    """
    Solve MILP problems to incrementally add dominating policies to a set
    """
    args = self.mdp
    
    opt, x = lpDual(**args)
    args['domPis'] = [x]
    args['consIdx'] = self.consIdx
    print opt
    printOccupancy(x)
    
    # iterate until no more dominating policies are found
    while True:
      opt, x = domPiMilp(**args)
      if opt == 0: break

      args['domPis'].append(x)
      print opt
      printOccupancy(x)
    
    return args['domPis']

  def envFeatureChanged(self, s0, s):
    # decide if s violates any environmental feature
    return any(s0[i] != s[i] for i in self.consIdx)
  
  # find marginalized state space
  def statesWithSameFeats(self, idx, value):
    return filter(lambda s: s[idx] == value, self.mdp['S'])

  def statesWithDifferentFeats(self, idx, value):
    return filter(lambda s: s[idx] != value, self.mdp['S'])

  def statesTransitToDifferentFeatures(self, idx, value):
    ret = []
    for s in self.mdp['S']:
      if s[idx] == value:
        for a in self.mdp['A']:
          for sp in self.mdp['S']:
            if self.mdp['T'](s, a, sp) > 0 and sp[idx] != value:
              ret.append((s, a))
              break
    return ret

def printOccupancy(x):
  for sa, occ in x.items():
    if occ > 0: print sa, occ

