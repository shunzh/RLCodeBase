import random
import numpy as np
import itertools

def convert(cmp, rewardSet, psi):
  """
  Convert a CMP instance to `args` that can be understood by the imlp solver 
  Note that R and psi are not provided in cmp and need to be supplemented later
  """
  ret = {}
  ret['S'] = cmp.getStates()
  ret['A'] = cmp.getPossibleActions(cmp.state) # assume state actions are available for all states
  def transition(s, a, sp):
    trans = cmp.getTransitionStatesAndProbs(s, a)
    trans = filter(lambda (state, prob): state == sp, trans)
    if len(trans) > 0: return trans[0][1]
    else: return 0
  ret['T'] = transition
  ret['R'] = rewardSet
  ret['s0'] = cmp.state
  ret['psi'] = psi
  return ret

def rewardConstruct(rocks):
  return lambda s, a: 1 if s in rocks else 0

def getRockDomain(size, numRocks, rewardCandNum, fixedRocks = False, randSeed = 0):
  """
  Return a domain with a number of random rocks.

  Return
  """
  random.seed(randSeed)
  ret = {}

  ret['S'] = [(x, y) for x in xrange(size) for y in xrange(size)]
  ret['A'] = [(1, -1), (1, 0), (1, 1)]
  def transit(s, a):
    loc = [s[0] + a[0], s[1] + a[1]]

    if loc[1] < 0: loc[1] = 0
    elif loc[1] >= size: loc[1] = size - 1
    
    return tuple(loc)
    
  ret['T'] = lambda s, a, sp: 1 if transit(s, a) == sp else 0

  ret['R'] = []
  if fixedRocks:
    possibleRocks = [(size - 1, 0), (size - 1, (size - 1) / 2), (size - 1, size - 1)]
    for i in xrange(len(possibleRocks)):
      ret['R'].append(rewardConstruct(possibleRocks[i:i+1]))
  else:
    for _ in xrange(rewardCandNum):
      # select rocks randomly from S
      rocks = np.random.permutation(ret['S'])[:numRocks]
      ret['R'].append(rewardConstruct(rocks))

  ret['s0'] = (0, size / 2)
  ret['psi'] = [1.0 / rewardCandNum] * rewardCandNum

  return ret

def getChainDomain(length):
  """
  A chain of states for debug
  """
  ret = {}

  ret['S'] = range(length)
  ret['A'] = [-1, 1]
  
  def transit(s, a):
    sp = s + a
    return sp

  ret['T'] = lambda s, a, sp: 1 if transit(s, a) == sp else 0
  ret['R'] = [lambda s, a: 1 if s == length - 1 and a == 1 else 0]
  ret['s0'] = length / 2
  ret['psi'] = [1]
  
  return ret

def getFactoredMDP(sSets, aSets, rFunc, tFunc, s0, terminal, gamma=1):
  ret = {}

  #ret['S'] = [s for s in itertools.product(*sSets)]
  ret['A'] = aSets
  # factored reward function
  #ret['r'] = lambda state, action: sum(r(s, a) for s, r in zip(state, rFunc))
  # nonfactored reward function
  ret['r'] = rFunc

  # t(s, a, s') = \prod t_i(s, a, s_i)
  transit = lambda state, action: tuple([t(state, action) for t in tFunc])
  
  # overriding this function depending on if sp is passed in
  #FIXME assume deterministic transitions for now to make the life easier!
  def transFunc(state, action, sp=None):
    if sp == None:
      return transit(state, action)
    else:
      return 1 if sp == transit(state, action) else 0

  ret['T'] = transFunc 
  ret['s0'] = s0
  ret['terminal'] = terminal
  ret['gamma'] = gamma

  #print transit(((2, 1), 0, 0, 1, 0, 1, 3), (1, 0))
  
  # construct the set of reachable states
  ret['S'] = []
  buffer = [s0]
  # stop when no new states are found by one-step transitions
  while len(buffer) > 0:
    # add the last batch to S
    ret['S'] += buffer
    newBuffer = []
    for s in buffer:
      if not terminal(s):
        for a in aSets:
          sp = transit(s, a)
          if not sp in ret['S'] and not sp in newBuffer: 
            newBuffer.append(sp)
    buffer = newBuffer

  return ret
