import util
from tabularNavigation import TabularNavigation, TabularNavigationToy
from tabularNavigationExp import experiment
import getopt
import sys
import tabularNavigationExp
import random
import config

if __name__ == '__main__':
  width = 10
  height = 10
  # the time step that the agent receives the response
  responseTime = 0
  horizon = height + width + 1
  rockNum = 3
  rewardCandNum = 5
  
  try:
    opts, args = getopt.getopt(sys.argv[1:], tabularNavigationExp.flags)
  except getopt.GetoptError:
    raise Exception('Unknown flag')
  for opt, arg in opts:
    if opt == '-t':
      rockNum = int(arg)
    elif opt == '-n':
      rewardCandNum = int(arg)
    elif opt == '-m':
      config.NUMBER_OF_RESPONSES = int(arg)
    elif opt == '-r':
      random.seed(int(arg))
  config.opts = '_'.join(map(str, [rockNum, rewardCandNum]))
  
  if rockNum == 0:
    Domain = TabularNavigationToy
  else:
    Domain = TabularNavigation
  
  def rewardGen(rewards, numerical): 
    def rewardFunc(s, a):
      if s in rewards:
        return numerical
      else:
        return 0
    return rewardFunc

  rewardSet = []
  if rockNum == 0:
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
  else:
    rocks = [(random.randint(0, width - 1), random.randint(0, height - 1)) for _ in xrange(10)]
    for candId in xrange(rewardCandNum):
      rewardSet.append(rewardGen(random.sample(rocks, rockNum), 1.0 / rockNum))

  initialPhi = [1.0 / rewardCandNum] * rewardCandNum

  terminalReward = util.Counter()

  experiment(Domain, width, height, responseTime, horizon, rewardCandNum, rewardSet, initialPhi, terminalReward)
