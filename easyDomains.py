import random
import numpy as np

def getRockDomain(size, numRocks, rewardCandNum, randSeed = 0):
  """
  Return a domain with a number of random rocks.

  Return
  """
  random.seed(randSeed)
  ret = {}

  ret['S'] = [(x, y) for x in xrange(size) for y in xrange(size)]
  ret['A'] = [(0, 1), (0, -1), (1, 0), (-1, 0)]
  def transition(s, a):
    loc = [s[0] + a[0], s[1] + a[1]]

    if loc[0] < 0: loc[0] = 0
    elif loc[0] >= size: loc[0] = size - 1
    
    if loc[1] < 0: loc[1] = 0
    elif loc[1] >= size: loc[1] = size - 1
    
    return tuple(loc)
  ret['T'] = transition

  ret['R'] = []
  for _ in xrange(rewardCandNum):
    # select rocks randomly from S
    rocks = np.random.permutation(S)[:numRocks]
    r = lambda s: 1 if s in rocks else 0
    ret['R'].append(r)

  ret['s0'] = (0, 0)
  ret['psi'] = [1.0 / rewardCandNum] * rewardCandNum]
