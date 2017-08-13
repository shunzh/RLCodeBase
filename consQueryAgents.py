from lp import lpDual

class ConsQueryAgent():
  def __init__(self, mdp, consIdx):
    """
    can't think of a class it should inherit..

    mdp: a factored mdp
    consIdx: specify a set of indices of features that the robot is not supposed to change without querying 
    """
    self.mdp = mdp
  
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
                             for idx in range(featLength)
                             for s in self.statesWithDifferentFeats(idx, s0[idx])}
    args['constraints'] = constraints
    args['positiveConstraints'] = {}
    rawOpt = lpDual(**args)

    irrFeats = []
    # solve the problem which only constrains one feature
    for idx in range(featLength):
      constraints = {(s, a): 0 for a in A
                               for s in self.statesWithDifferentFeats(idx, s0[idx])}
      args['constraints'] = {}
      args['positiveConstraints'] = constraints
      opt = lpDual(**args)
      
      if opt <= rawOpt: irrFeats.append(idx)
    
    return irrFeats
  
  # find marginalized state space
  def statesWithSameFeats(self, idx, value):
    return filter(lambda s: s[idx] == value, self.mdp['S'])
  def statesWithDifferentFeats(self, idx, value):
    return filter(lambda s: s[idx] != value, self.mdp['S'])
