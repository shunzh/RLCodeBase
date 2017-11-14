from lp import lpDual, computeValue
import pprint
from util import powerset
import copy
import numpy
import itertools

VAR = 0
# not considering reversible features in this branch
NONREVERSED = 1

class ConsQueryAgent():
  """
  Find queries in constraint-uncertain mdps.
  """
  def __init__(self, mdp, consStates, constrainHuman=False):
    """
    can't think of a class it should inherit..

    mdp: a factored mdp
    consSets: the set of environmental feature indices
    """
    self.mdp = mdp

    # indices of constraints
    self.consStates = consStates
    self.consIndices = range(len(consStates))
    
    # derive different definition of MR
    self.constrainHuman = constrainHuman

    self.allCons = [(VAR, _) for _ in self.consIndices]
  
  def findConstrainedOptPi(self, activeCons):
    mdp = copy.copy(self.mdp)

    mdp['constraints'] = self.constructConstraints(activeCons, mdp)
    opt, x = lpDual(**mdp)

    return x

  def findRelevantFeaturesAndDomPis(self):
    """
    Incrementally add dominating policies to a set
    """
    beta = [] # rules to keep
    dominatingPolicies = {}

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

      x = self.findConstrainedOptPi(activeCons)

      dominatingPolicies[activeCons] = x

      # check violated constraints
      if x == {}:
        violatedCons = ()
      else:
        violatedCons = self.findViolatedConstraints(x)

      # beta records that we would not enforce activeCons and relax occupiedFeats in the future
      beta.append((set(activeCons), set(violatedCons)))

      allCons.update(violatedCons)

      allConsPowerset = set(powerset(allCons))

      #print 'beta', beta
      #print 'allCons', allCons
    
    domPis = []
    for pi in dominatingPolicies.values():
      if pi not in domPis: domPis.append(pi)
      
    print 'rel cons', allCons, 'num of domPis', len(domPis)
    return allCons, domPis

  def findMinimaxRegretConstraintQBruteForce(self, k, relFeats, domPis):
    mrs = {}

    if len(relFeats) < k:
      # we have a chance to ask about all of them!
      return tuple(relFeats)
    else:
      for q in itertools.combinations(relFeats, k):
        mr, advPi = self.findMRAdvPi(q, relFeats, domPis, k)
        mrs[q] = mr

      if mrs == {}:
        mmq = () # no need to ask anything
      else:
        mmq = min(mrs.keys(), key=lambda _: mrs[_])
    
      return mmq

  def findMinimaxRegretConstraintQ(self, k, relFeats, domPis, scopeHeu=True, filterHeu=True):
    """
    Finding a minimax k-element constraint query.
    
    Use pruning if pruning=True, otherwise brute force.
    """
    if len(relFeats) < k:
      # we have a chance to ask about all of them!
      return tuple(relFeats)

    # candidate queries and their violated constraints
    candQVCs = {}
    mrs = {}

    # first query to consider
    q = self.findChaindAdvConstraintQ(k, relFeats, domPis)
    
    if len(q) < k: return q # mr = 0 in this case

    allCons = set()
    allCons.update(q)

    if scopeHeu:
      allConsPowerset = set(itertools.combinations(allCons, k))
    else:
      allConsPowerset = set(itertools.combinations(relFeats, k))

    qChecked = []

    while True:
      qToConsider = allConsPowerset.difference(qChecked)

      if len(qToConsider) == 0: break

      # find the subset with the smallest size
      q = qToConsider.pop()
      qChecked.append(q)

      print 'q', q

      # check the pruning condition
      dominatedQ = False
      if filterHeu:
        for candQ in candQVCs.keys():
          if set(q).intersection(candQVCs[candQ]).issubset(candQ):
            dominatedQ = True
            break
        if dominatedQ:
          print q, 'is dominated'
          continue

      mr, advPi = self.findMRAdvPi(q, relFeats, domPis, k)

      candQVCs[q] = self.findViolatedConstraints(advPi)
      allCons.update(candQVCs[q])
      if scopeHeu:
        allConsPowerset = set(itertools.combinations(allCons, k))
      # allConsPowerset is consistent (all k-subsets) if not scope. no need to update. 

      mrs[q] = mr
      
    mmq = min(mrs.keys(), key=lambda _: mrs[_])
  
    return mmq

  def findChaindAdvConstraintQ(self, k, relFeats, domPis):
    q = set()
    while len(q) <= k:
      sizeOfQ = len(q)

      mr, advPi = self.findMRAdvPi(q, relFeats, domPis, k)
      q = q.union(self.findViolatedConstraints(advPi))
      
      if len(q) == sizeOfQ: break # no more constraints to add
    
    # may exceed k constraints. return the first k constraints only
    mmq = list(q)[:k]
    return mmq

  def findRelevantRandomConstraintQ(self, k, relFeats):
    if len(relFeats) >= k:
      q = numpy.random.choice(relFeats, k)
    else:
      # no more than k relevant features
      q = relFeats
    
    return q
 
  def findRandomConstraintQ(self, k):
    if len(self.consIndices) >= k:
      q = numpy.random.choice(self.consIndices, k)
    else:
      # no more than k constraints, should not design exp in this way though
      q = self.consIndices
    
    q = [(VAR, _) for _ in q]
    
    return q
  
  def findRegret(self, q, violableCons):
    """
    A utility function that finds regret given the true violable constraints
    """
    consRobotCanViolate = set(q).intersection(violableCons)
    rInvarCons = set(self.allCons).difference(consRobotCanViolate)
    robotPi = self.findConstrainedOptPi(rInvarCons)
    
    hInvarCons = set(self.allCons).difference(violableCons)
    humanPi = self.findConstrainedOptPi(hInvarCons)
    
    hValue = self.computeValue(humanPi)
    rValue = self.computeValue(robotPi)
    
    return hValue - rValue

  def findRobotDomPis(self, q, relFeats, domPis):
    """
    Find the set of dominating policies adoptable by the robot.
    """
    invarFeats = set(relFeats).difference(q)
    pis = []

    for rPi in domPis:
      if self.piSatisfiesCons(rPi, invarFeats):
        pis.append(rPi)

    return pis

  def findMRAdvPi(self, q, relFeats, domPis, k):
    """
    Find the adversarial policy given q and domPis
    
    Now searching over all dominating policies, maybe take some time.. can use MILP instead?
    """
    maxRegret = 0
    maxValues = None
    advPi = None

    for pi in domPis:
      humanViolated = self.findViolatedConstraints(pi)
      humanValue = self.computeValue(pi)

      if self.constrainHuman and len(humanViolated) > k:
        # we do not consider the case where the human's optimal policy violates more than k constraints
        # unfair to compare.
        continue

      # intersection of q and constraints violated by pi
      consRobotCanViolate = set(q).intersection(humanViolated)
      
      # the robot's optimal policy given the constraints above
      invarFeats = set(relFeats).difference(consRobotCanViolate)
      
      robotValue = 0
      for rPi in domPis:
        if self.piSatisfiesCons(rPi, invarFeats):
          rValue = self.computeValue(rPi)
          if rValue > robotValue:
            robotValue = rValue

      regret = humanValue - robotValue
      
      assert regret >= 0, 'regret is %f' % regret
      if regret > maxRegret or (regret == maxRegret and advPi == None):
        maxRegret = regret
        advPi = pi
        maxValues = (humanValue, robotValue) # not used, just for debugging
  
    # even with constrainHuman, the non-constraint-violating policy is in \Gamma
    assert advPi != None
    print maxValues, maxRegret
    return maxRegret, advPi

  def constructConstraints(self, cons, mdp):
    """
    Construct set of constraint equations by the specification in cons
    """
    constraints = {}
    for con in cons:
      consType, consIdx = con
      if consType == VAR:
        constraints.update({(s, a): 0 for a in mdp['A']
                                      for s in self.consStates[consIdx]})
      elif consType == NONREVERSED:
        constraints.update({(s, a): 0 for a in mdp['A']
                                      for s in self.statesWithDifferentFeats(consIdx, mdp)
                                      if mdp['terminal'](s)})
      else: raise Exception('unknown constraint type')

    return constraints

  def computeValue(self, x):
    return computeValue(x, self.mdp['r'], self.mdp['S'], self.mdp['A'])

  def piSatisfiesCons(self, x, cons):
    violatedCons = self.findViolatedConstraints(x)
    return set(cons).isdisjoint(set(violatedCons))

  def findViolatedConstraints(self, x):
    # set of changed features
    var = set()
    
    for idx in self.consIndices:
      # states violated by idx
      for s, a in x.keys():
        if any(x[s, a] > 0 for a in self.mdp['A']) and s in self.consStates[idx]:
          var.add(idx)
    
    return set([(VAR, idx) for idx in var])
    
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
