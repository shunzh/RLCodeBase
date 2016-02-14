import util
from tabularNavigation import TabularNavigation
from tabularNavigationExp import experiment
import getopt
import sys
import tabularNavigationExp
import random
from AugmentedCMP import AugmentedCMP

if __name__ == '__main__':
  width = 21
  height = 21
  # the time step that the agent receives the response
  responseTime = 10
  horizon = 40
  
  try:
    opts, args = getopt.getopt(sys.argv[1:], tabularNavigationExp.flags)
  except getopt.GetoptError:
    raise Exception('Unknown flag')
  for opt, arg in opts:
    if opt == '-t':
      rockType = arg
    elif opt == '-r':
      random.seed(int(arg))

  rocks = [(1, 0), (0, 1),
           (width - 2, 0), (width - 1, 1)]
  rewardCandNum = 4
  rewards = []
  for candId in xrange(rewardCandNum):
    reward = util.Counter()
    reward[rocks[candId]] = 1
    rewards.append(reward)

  initialPhi = [1] * len(rocks)
  initialPhi = map(lambda _: _ / sum(initialPhi), initialPhi)

  Domain = AugmentedCMP(TabularNavigation, rewardSet, initialPhi, queryType, gamma, 1)