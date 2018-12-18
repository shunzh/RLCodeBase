import random
import numpy as np
import itertools
import copy

class SimpleMDP:
  def __init__(self, S=[], A=[], T=None, r=None, s0=None, terminal=lambda _: False, gamma=1, psi=[1]):
    """
    Most methods do component assignment separately, so setting dummy default values.
    """
    self.S = S
    self.A = A
    self.T = T
    self.r = r
    self.s0 = s0
    self.terminal = terminal
    self.gamma = gamma
    self.psi = psi

def convert(cmp, rewardSet, psi):
  """
  Convert a CMP instance to `args` that can be understood by the imlp solver 
  Note that R and psi are not provided in cmp and need to be supplemented later
  """
  ret = SimpleMDP()
  ret.S = cmp.getStates()
  ret.A = cmp.getPossibleActions(cmp.state) # assume state actions are available for all states
  def transition(s, a, sp):
    trans = cmp.getTransitionStatesAndProbs(s, a)
    trans = filter(lambda (state, prob): state == sp, trans)
    if len(trans) > 0: return trans[0][1]
    else: return 0
  ret.T = transition
  ret.r = rewardSet
  ret.s0 = cmp.state
  ret.psi = psi

  # FIXME should simple MDP support psi??
  return ret

def occupancyAdd(mdp, pi, piP, scalar):
  """
  pi := pi + piP * scalar
  """
  for s in mdp.S:
    for a in mdp.A:
      if (s, a) in piP.keys():
        # otherwise there is nothing to add here
        if (s, a) in pi.keys():
          pi[s, a] += piP[s, a] * scalar
        else:
          pi[s, a] = piP[s, a] * scalar

def policyToOccupancyFromAllS0(mdp, piOcc):
  """
  DUMMY?

  given the policy pi[s, a], compute the occupancy measure
  basically return inv(I - \gamma P)
  """
  # make sure all states have zero occupancy (alpha should take care of this)
  # otherwise not invertible
  N = len(mdp.S)
  
  pi = {}
  sToSpProb = {}
  
  # convert occupancy to policy
  for s in mdp.S:
    for a in mdp.A:
      pi[s, a] = piOcc[s, a] / sum(piOcc[s, ap] for ap in mdp.A)

  for s in mdp.S:
    for sp in mdp.S:
      sToSpProb[s, sp] = sum(pi[s, a] * mdp.T(s, a, sp) for a in mdp.A)
  
  P = np.matrix([[sToSpProb[s, sp] for sp in mdp.S] for s in mdp.S])

  occ = np.linalg.inv(np.eye(N) - mdp.gamma * P)
  
  return {s: {sp: occ[mdp.S.index(s)][mdp.S.index(sp)] for sp in mdp.S} for s in mdp.S}
 
  
def rewardConstruct(rocks):
  return lambda s, a: 1 if s in rocks else 0

def getRockDomain(size, numRocks, rewardCandNum, fixedRocks = False, randSeed = 0):
  """
  Return a domain with a number of random rocks.

  Return
  """
  random.seed(randSeed)
  ret = SimpleMDP()

  ret.S = [(x, y) for x in xrange(size) for y in xrange(size)]
  ret.A = [(1, -1), (1, 0), (1, 1)]
  def transit(s, a):
    loc = [s[0] + a[0], s[1] + a[1]]

    if loc[1] < 0: loc[1] = 0
    elif loc[1] >= size: loc[1] = size - 1
    
    return tuple(loc)
    
  ret.T = lambda s, a, sp: 1 if transit(s, a) == sp else 0

  ret.R = []
  if fixedRocks:
    possibleRocks = [(size - 1, 0), (size - 1, (size - 1) / 2), (size - 1, size - 1)]
    for i in xrange(len(possibleRocks)):
      ret.R.append(rewardConstruct(possibleRocks[i:i+1]))
  else:
    for _ in xrange(rewardCandNum):
      # select rocks randomly from S
      rocks = np.random.permutation(ret['S'])[:numRocks]
      ret.R.append(rewardConstruct(rocks))

  ret.s0 = (0, size / 2)
  ret.psi = [1.0 / rewardCandNum] * rewardCandNum

  return ret

def getChainDomain(length):
  """
  A chain of states for debug
  """
  ret = SimpleMDP()

  ret.S = range(length)
  ret.A = [-1, 1]
  
  def transit(s, a):
    sp = s + a
    return sp

  ret.T = lambda s, a, sp: 1 if transit(s, a) == sp else 0
  ret.R = [lambda s, a: 1 if s == length - 1 and a == 1 else 0]
  ret.s0 = length / 2
  ret.psi = [1]
  
  return ret

def getFactoredMDP(sSets, aSets, rFunc, tFunc, s0, gamma=1, terminal=lambda s: False):
  ret = SimpleMDP()

  #ret['S'] = [s for s in itertools.product(*sSets)]
  ret.A = aSets
  # factored reward function
  #ret['r'] = lambda state, action: sum(r(s, a) for s, r in zip(state, rFunc))
  # nonfactored reward function
  ret.r = rFunc

  # t(s, a, s') = \prod t_i(s, a, s_i)
  transit = lambda state, action: tuple([t(state, action) for t in tFunc])
  
  # overriding this function depending on if sp is passed in
  #FIXME assume deterministic transitions for now to make the life easier!
  def transFunc(state, action, sp=None):
    if sp == None:
      return transit(state, action)
    else:
      return 1 if sp == transit(state, action) else 0

  ret.T = transFunc 
  ret.alpha = lambda s: s == s0 # assuming there is only one starting state
  ret.terminal = terminal
  ret.gamma = gamma

  #print transit(((2, 1), 0, 0, 1, 0, 1, 3), (1, 0))
  
  # construct the set of reachable states
  ret.S = []
  buffer = [s0]
  # stop when no new states are found by one-step transitions
  while len(buffer) > 0:
    # add the last batch to S
    ret.S += buffer
    newBuffer = []
    for s in buffer:
      if not terminal(s):
        for a in aSets:
          sp = transit(s, a)
          if not sp in ret.S and not sp in newBuffer: 
            newBuffer.append(sp)
    buffer = newBuffer

  return ret


def randMDP(states, actions, rewardSparsity):
  """
  return an MDP with specified num of states and num of actions, with deterministic transitions to randomly-selected states
  
  """
  rDict = {}
  tDict = {}
  for s in states:
    # the reward is 1 w.p. rewardSparsity and 0 otherwise
    rDict[s] = random.random() < rewardSparsity

    for a in actions:
      # the transition state is randomly chosen 
      tDict[s, a] = random.choice(states)
  
  # let rewards defined on states, that is, r(s, a) = r(s, a') for all s, a, a'
  R = lambda s, a: rDict[s]
  # for now, consider deterministic transition functions
  T = lambda s, a, sp: tDict[s, a] == sp
  
  s0 = states[0]
  
  gamma = .9
  
  # for now assume no terminal states
  terminal = lambda s: False
  
  # return the mdp in a tuple
  return SimpleMDP(states, actions, T, R, s0, terminal, gamma)
