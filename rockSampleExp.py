import util
from tabularNavigation import RockCollection
from tabularNavigationExp import experiment
import getopt
import sys
import tabularNavigationExp
import random
import config
import numpy

if __name__ == '__main__':
  width = 10
  height = 10
  # the time step that the agent receives the response
  responseTime = 0
  horizon = height + width + 1
  rockNum = 10
  rewardCandNum = 10
  rewardVar = 0
  
  try:
    opts, args = getopt.getopt(sys.argv[1:], tabularNavigationExp.flags)
  except getopt.GetoptError:
    raise Exception('Unknown flag')
  for opt, arg in opts:
    if opt == '-t':
      config.TRAJECTORY_LENGTH = int(arg)
    elif opt == '-n':
      config.NUMBER_OF_QUERIES = int(arg)
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
  r1 = lambda s, a: s == (4, 1)
  r2 = lambda s, a: s == (4, 0)
  r3 = lambda s, a: s == (2, 0)
  r4 = lambda s, a: s == (3, 0)

  l1 = lambda s, a: s == (0, 2)
  l2 = lambda s, a: s == (0, 3)
  l3 = lambda s, a: s == (0, 4)
  l4 = lambda s, a: s == (0, 5)
  rewardSet = [r1, r2, r3, r4, l1, l2, l3, l4]

  # THE GENERAL CASE
  """
  # rocks can show up in any state
  rocks = cmp.getStates()
  # sample rockNum rocks from the state space
  for candId in xrange(rewardCandNum):
    # we have some more valuable rocks
    reward = random.random() * rewardVar + (1 - 1.0 * rewardVar / 2)
    rewardSet.append(rewardGen(random.sample(rocks, rockNum), reward))

  initialPhi = [1.0 / rewardCandNum] * rewardCandNum

  experiment(cmp, rewardSet, initialPhi)