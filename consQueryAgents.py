from lp import lpDual, domPiMilp, decomposePiLP, computeValue
import pprint
from util import powerset
import easyDomains
import copy
import util

VAR = 0
NONREVERSED = 1

class ConsQueryAgent():
  """
  Find queries in constraint-uncertain mdps. May formulate constraints as negative rewards.

  TODO only implementing some auxiliary functions. 
  """
  def __init__(self, sSets, aSets, rFunc, tFunc, s0, terminal, gamma, consSets, dependentSets):
    """
    can't think of a class it should inherit..

    mdp: a factored mdp
    consSets: the set of environmental feature indices
    """
    self.sSets = sSets # set of possible values of features for all features
    self.aSets = aSets
    self.rFunc = rFunc
    self.tFunc = tFunc
    self.s0 = s0
    self.terminal = terminal
    self.gamma = gamma
    
    self.transit = lambda state, action: tuple([t(state, action) for t in tFunc])

    # indices of constraints
    self.consSets = consSets
    self.consSetsSize = len(consSets)
    
    # set of features that should not be masked
    self.dependentSets = dependentSets
  
  def findRelevantFeatures(self):
    """
    Incrementally add dominating policies to a set
    """
    beta = [] # rules to keep

    allCons = set()
    allConsPowerset = set(powerset(allCons))
    subsetsConsidered = []
    
    # iterate until no more dominating policies are found
    while True:
      subsetsToConsider = allConsPowerset.difference(subsetsConsidered)

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
      if skipThisCons:
        continue

      maskIdx = [_ for _ in self.consSets if not (VAR, _) in activeCons
                                         and not (NONREVERSED, _) in activeCons
                                         and not _ in self.dependentSets]
      mdp = self.constructReducedFactoredMDP(maskIdx)

      mdp['constraints'] = self.constructConstraints(activeCons, mdp)
      opt, x = lpDual(**mdp)
      print 'occ'
      printOccSA(x)
      
      x = self.constructRawPolicy(x, maskIdx)
      print 'rawocc'
      printOccSA(x)
      # check violated constraints
      if x == {}:
        violatedCons = ()
      else:
        violatedCons = self.findViolatedConstraints(x)

      beta.append((set(activeCons), set(violatedCons)))

      # beta records that we would not enforce activeCons and relax occupiedFeats in the future
      for idx in self.consSets:
        if (NONREVERSED, idx) in violatedCons:
          allCons.add((NONREVERSED, idx))
        elif (NONREVERSED, idx) in activeCons and (VAR, idx) in violatedCons:
          allCons.add((VAR, idx))

      allConsPowerset = set(powerset(allCons))

      print 'beta', beta
      print 'allCons', allCons
      print opt
    
    return allCons

  def constructReducedFactoredMDP(self, maskIdx):
    sSets = copy.copy(self.sSets)
    tFunc = copy.copy(self.tFunc)

    for idx in maskIdx:
      sSets[idx] = [self.s0[idx]]
      tFunc[idx] = lambda s, a: s[idx]

    return easyDomains.getFactoredMDP(sSets, self.aSets, self.rFunc, tFunc, self.s0, self.terminal, self.gamma)
  
  # FIXME assuming deterministic transition
  def constructRawPolicy(self, x, maskIdx):
    newX = util.Counter()
    newS = self.s0
    mask = lambda state: tuple([self.s0[idx] if idx in maskIdx else state[idx] for idx in range(len(self.s0))])

    while True:
      s = newS
      # add the last batch to S
      for a in self.aSets:
        if (mask(s), a) in x.keys() and x[(mask(s), a)] > 0:
          newX[(s, a)] = x[mask(s), a]
          newS = self.transit(s, a)
          break
      if self.terminal(s) or newS == s: break
    return newX
  
  def findRelevantFeatsUsingHeu(self):
    """
    FIXME not sure whether we are going to include this algorithm. not updated.
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

  def constructConstraints(self, cons, mdp):
    """
    Construct set of constraint equations by the specification in cons
    """
    constraints = {}
    for con in cons:
      consType, consIdx = con
      if consType == VAR:
        constraints.update({(s, a): 0 for a in mdp['A']
                                      for s in self.statesWithDifferentFeats(consIdx, mdp)})
      elif consType == NONREVERSED:
        constraints.update({(s, a): 0 for a in mdp['A']
                                      for s in self.statesWithDifferentFeats(consIdx, mdp)
                                      if mdp['terminal'](s)})
      else: raise Exception('unknown constraint type')

    return constraints

  def findViolatedConstraints(self, x):
    # set of changed features
    var = set()
    # set of features that are different from the initial value at time step T
    notReversed = set()
    
    for idx in self.consSets:
      # states violated by idx
      for s, a in x.keys():
        if s[idx] != self.s0[idx] and any(x[s, a] > 0 for a in self.aSets):
          var.add(idx)
          if self.terminal(s):
            notReversed.add(idx)

    return set([(VAR, idx) for idx in var] + [(NONREVERSED, idx) for idx in notReversed])
    
  def statesWithDifferentFeats(self, idx, mdp):
    return filter(lambda s: s[idx] != mdp['s0'][idx], mdp['S'])

  # FIXME remove or not? only used by depreciated methods
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
