import random
import numpy as np

def getRockDomain(size, numRocks, rewardCandNum, fixedRocks = False, randSeed = 0):
  """
  Return a domain with a number of random rocks.

  Return
  """
  random.seed(randSeed)
  ret = {}

  ret['S'] = [(x, y) for x in xrange(size) for y in xrange(size)]
  ret['A'] = [(-1, 1), (0, 1), (1, 1)]
  def transition(s, a, sp):
    loc = [s[0] + a[0], s[1] + a[1]]

    if loc[0] < 0: loc[0] = 0
    elif loc[0] >= size: loc[0] = size - 1
    
    if loc[1] < 0: loc[1] = 0
    elif loc[1] >= size: loc[1] = size - 1
    
    if tuple(loc) == sp: return 1
    else: return 0
  ret['T'] = transition

  ret['R'] = []
  if fixedRocks:
    possibleRocks = [(0, size - 1),\
                     ((size - 1) / 3, size - 1),\
                     ((size - 1) * 2 / 3, size - 1),\
                     (size - 1, size - 1)]
    for i in xrange(4):
      r = lambda s, a: 1 if s == possibleRocks[i] else 0
      ret['R'].append(r)
  else:
    for _ in xrange(rewardCandNum):
      # select rocks randomly from S
      rocks = np.random.permutation(ret['S'])[:numRocks]
      r = lambda s, a: 1 if s in rocks else 0
      ret['R'].append(r)

  ret['s0'] = (size / 2, 0)
  ret['psi'] = [1.0 / rewardCandNum] * rewardCandNum

  return ret