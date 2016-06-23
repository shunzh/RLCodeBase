import random
import numpy as np

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