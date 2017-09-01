from lp import lpDual, domPiMilp, decomposePiLP, computeValue
import pprint
from util import powerset

class ConsQueryAgent():
  """
  Find queries in constraint-uncertain mdps. May formulate constraints as negative rewards.

  TODO only implementing some auxiliary functions. 
  """
  def __init__(self, mdp, consSets):
    """
    can't think of a class it should inherit..

    consSets: a set of possible constraints. consSets[i] is a set of states.
              if i-th constraint is active, then the robot cannot visit consSets[i]
    """
    self.mdp = mdp
    self.consSets = consSets
    self.consSetsSize = len(consSets)
  
  def findIrrelevantFeats(self):
    """
    DEPRECIATED. The criterion is too strong. unlikely to rule out any features.
    TODO. still using consIdx rather than consSets. need updates.

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
  
  def findRelevantFeatures(self):
    """
    Solve MILP problems to incrementally add dominating policies to a set
    """
    args = self.mdp
    s0 = args['s0']
    A = args['A']

    beta = [] # rules to keep
    relFeats = set()
    
    relFeatPowerSet = set(powerset(relFeats))
    subsetsConsidered = []
    
    # iterate until no more dominating policies are found
    while True:
      subsetsToConsider = relFeatPowerSet.difference(subsetsConsidered)

      if len(subsetsToConsider) == 0: break

      # find the subset with the smallest size
      activeCons = min(subsetsToConsider, key=lambda _: len(_))
      subsetsConsidered.append(activeCons)
      
      skipThisCons = False
      for enf, relax in beta:
        if enf.issubset(activeCons) and len(relax.intersection(activeCons)) == 0:
          # this subset can be ignored
          skipThisCons = True
          break
      if skipThisCons: continue

      args['constraints'] = {(s, a): 0 for a in A
                             for idx in activeCons
                             for s in self.consSets[idx]}
      opt, x = lpDual(**args)
      
      # check how many features are violated
      # x is {} if no feasible features
      occupiedFeats = set()
      for sa, occ in x.items():
        if occ > 0:
          s, a = sa
          for idx in range(self.consSetsSize):
            if s in self.consSets[idx]: 
              occupiedFeats.add(idx)

      # beta records that we would not enforce activeCons and relax occupiedFeats in the future
      beta.append((set(activeCons), occupiedFeats))

      relFeats = relFeats.union(occupiedFeats)
      relFeatPowerSet = set(powerset(relFeats))

      print 'beta', beta
      #print opt
      #printOccupancy(x)
    
    return list(relFeats)

  def findRelevantFeatsUsingHeu(self):
    """
    FIXME not updated.
    This finds a superset of all relevant features
    """
    args = self.mdp
    S = args['S']
    A = args['A']

    relFeats = set()
    
    # find the policy with all constraints relaxed
    bestOpt, bestX = lpDual(**args)
    bestXOccupied = findOccupiedStates(bestX)
    for idx in range(self.consSetsSize):
      for s in bestXOccupied:
        if s in self.consSets[idx]: 
          relFeats.add(idx)
          break
    print 'best x'
    printOccSA(bestX)
    print relFeats

    # find the policy with all constraints enforced
    args['constraints'] = {(s, a): 0 for a in A
                           for s in S
                           for consSet in self.consSets
                           if s in consSet}
    rawOpt, rawX = lpDual(**args)

    while True:
      # the optimal policy that has to change some features other than known relevant features
      args['constraints'] = {}
      args['positiveConstraints'] = [(s, a) for a in A
                             for idx in range(self.consSetsSize)
                             if not idx in relFeats
                             for s in self.consSets[idx]]
      opt, x = lpDual(**args)
      
      if x == {}: break # such pi does not exist

      sigma, y = decomposePiLP(S, A, args['T'], args['s0'], args['terminal'], bestX, x)
      
      if sigma == 1: break # y == bestX

      print 'sigma', sigma
      print computeValue(y, args['r'], S, A) / (1 - sigma), rawOpt
      if computeValue(y, args['r'], S, A) / (1 - sigma) <= rawOpt: break

      yOccupied = findOccupiedStates(y)
      for idx in range(self.consSetsSize):
        for s in yOccupied:
          if s in self.consSets[idx]: 
            relFeats.add(idx)
            break

      print relFeats
   
    return relFeats

  def findRelevantFeatsBruteForce(self):
    """
    Baseline. we can find all relevant features by enumerating all dominating policies
    """
    args = self.mdp
    A = args['A']
    
    relFeats = set()
 
    for activeCons in powerset(range(self.consSetsSize)):
      print activeCons
      # solve the raw problem
      args['constraints'] = {(s, a): 0 for a in A
                             for idx in activeCons
                             for s in self.consSets[idx]}
      #print args['constraints']
      opt, x = lpDual(**args)
      #print opt
      #printOccSA(x)

      for idx in range(self.consSetsSize):
        for sa, occ in x.items():
          if occ > 0:
            s, a = sa
            if s in self.consSets[idx]: 
              relFeats.add(idx)
              #print 'add', idx

    return list(relFeats)

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

def findOccupiedStates(x):
  states = set()
  for sa, occ in x.items():
    if occ > 0:
      s, a = sa
      states.add(s)
  
  return states

def printOccSA(x):
  for sa, occ in x.items():
    if occ > 0: print sa, occ