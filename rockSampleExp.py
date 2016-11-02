import util
from tabularNavigation import TabularNavigation, TabularNavigationToy,\
  RockCollection
from tabularNavigationExp import experiment
import getopt
import sys
import tabularNavigationExp
import random
import config

if __name__ == '__main__':
  width = 20
  height = 20
  # the time step that the agent receives the response
  responseTime = 0
  horizon = height + width + 1
  rockNum = 10
  rewardCandNum = 5
  
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
    elif opt == '-x':
      config.NUMBER_OF_QUERIES = int(arg)
    elif opt == '-y':
      rockNum = int(arg)
    elif opt == '-r':
      random.seed(int(arg))
  config.opts = '_'.join(map(str, [rewardCandNum, config.NUMBER_OF_QUERIES, config.NUMBER_OF_RESPONSES, config.TRAJECTORY_LENGTH]))
  
  def rewardGen(rewards, numerical): 
    def rewardFunc(s, a):
      if s in rewards:
        return numerical
      else:
        return 0
    return rewardFunc

  Domain = RockCollection
  terminalReward = util.Counter()
  cmp = Domain(responseTime, width, height, horizon = horizon, terminalReward = terminalReward)

  rewardSet = []
  if rockNum == 0:
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
    r1 = lambda s, a: s == (4, 0)
    r2 = lambda s, a: s == (0, 4)
    r3 = lambda s, a: s == (2, 2)
    rewardSet = [r1, r2, r3]
  else:
    # rocks can show up in any state
    rocks = cmp.getStates()
    # sample rockNum rocks from the state space
    for candId in xrange(rewardCandNum):
      rewardSet.append(rewardGen(random.sample(rocks, rockNum), 1.0 / rockNum))

  initialPhi = [1.0 / rewardCandNum] * rewardCandNum

  experiment(cmp, rewardSet, initialPhi)