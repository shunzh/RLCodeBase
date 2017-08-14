from lp import lpDual
import pprint

class ConsQueryAgent():
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
      constraints = [(s, a) for a in A
                            for s in self.statesWithDifferentFeats(idx, s0[idx])]
      args['constraints'] = {}
      args['positiveConstraints'] = constraints
      opt, occ = lpDual(**args)
      print opt
      for s, prob in occ.items():
        if prob > 0: print s, prob
      
      if opt <= rawOpt: irrFeats.append(idx)
    
    return irrFeats
  
  # find marginalized state space
  def statesWithSameFeats(self, idx, value):
    return filter(lambda s: s[idx] == value, self.mdp['S'])
  def statesWithDifferentFeats(self, idx, value):
    return filter(lambda s: s[idx] != value, self.mdp['S'])
