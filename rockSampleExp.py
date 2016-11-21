import util
from tabularNavigation import RockCollectionDiagonal, RockCollection
from tabularNavigationExp import experiment
import getopt
import sys
import tabularNavigationExp
import random
import config
import numpy

if __name__ == '__main__':
  width = 5
  height = 30
  # the time step that the agent receives the response
  responseTime = 0
  horizon = height + 1
  rockNum = 20#40
  rewardCandNum = 5#10
  rewardVar = 1
  
  try:
    opts, args = getopt.getopt(sys.argv[1:], tabularNavigationExp.flags)
  except getopt.GetoptError:
    raise Exception('Unknown flag')
  for opt, arg in opts:
    if opt == '-t':
      config.TRAJECTORY_LENGTH = int(arg)
    elif opt == '-n':
      rewardCandNum = int(arg)
    elif opt == '-k':
      config.NUMBER_OF_RESPONSES = int(arg)
    elif opt == '-y':
      rewardVar = int(arg)
    elif opt == '-r':
      random.seed(int(arg))
      numpy.random.seed(int(arg))
  config.opts = '_'.join(map(str, [rewardCandNum, config.NUMBER_OF_QUERIES, config.NUMBER_OF_RESPONSES, rewardVar]))
  
  def rewardGen(rewards, numerical): 
    def rewardFunc(s, a):
      if s in rewards:
        return numerical
      else:
        return 0
    return rewardFunc
  
  def reward2Gen(rewards1, rewards2, numerical1, numerical2):
    def rewardFunc(s, a):
      if s in rewards1:
        return numerical1
      elif s in rewards2:
        return numerical2
      else:
        return 0
    return rewardFunc
 
  Domain = RockCollection
  terminalReward = util.Counter()
  cmp = Domain(responseTime, width, height, horizon = horizon, terminalReward = terminalReward)

  rewardSet = []
  
  # special cases for debugging
  """
  # use rockNum == 0 to represent a test case
  rewardCandNum = 3
  def r1(s, a):
    if s == (0, 0) and a == (1, 0): return 0.9
    elif s == (0, 1) and a == (0, 1): return 0.6
    else: return 0
  def r2(s, a):
    if s == (0, 1) and a == (1, 0): return 1
    elif s == (0, 1) and a == (0, 1): return 0.6
    else: return 0
  def r3(s, a):
    if s == (0, 0) and a == (1, 0): return 0.45
    elif s == (0, 1) and a == (1, 0): return 0.5
    elif s == (0, 1) and a == (0, 1): return 0.6
    else: return 0
  rewardSet = [r1, r2, r3]
  """
  """
  r1 = lambda s, a: s == (2, 10)
  r2 = lambda s, a: s == (1, 10)
  r3 = lambda s, a: s == (3, 10)
  rewardSet = [r1, r2, r3]

  # THE GENERAL CASE
  """
  
  # don't set rewards on terminal states (where y == height - 1)
  rocks = random.sample(filter(lambda (x, y): y < cmp.height - 1, cmp.getStates()), rockNum)

  if rewardVar == 1:
    rewardCandBonus = [1] * rewardCandNum
  elif rewardVar == 2:
    rewardCandBonus = [2] * (rewardCandNum / 2) + [1] * (rewardCandNum - rewardCandNum / 2)
  elif rewardVar == 3:
    rewardCandBonus = [5] + [1] * (rewardCandNum - 1)
  else:
    raise Exception('unknown rewardVar')

  # sample rockNum rocks from the state space
  for candId in xrange(rewardCandNum):
    # we have some more valuable rocks
    bonus = random.sample(rocks, rockNum / rewardCandNum)
    pits = [_ for _ in rocks if not _ in bonus] # pits = rocks \ bonus
    rewardSet.append(reward2Gen(bonus, pits, rewardCandBonus[candId], -10))

  initialPhi = [1.0 / rewardCandNum] * rewardCandNum

  experiment(cmp, rewardSet, initialPhi)
