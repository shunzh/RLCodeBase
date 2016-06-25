import util
from tabularNavigation import TabularNavigation
from tabularNavigationExp import experiment
import getopt
import sys
import tabularNavigationExp
import random

if __name__ == '__main__':
  width = 21
  height = 21
  # the time step that the agent receives the response
  responseTime = 0
  horizon = 20
  rockType = 'default'
  extra = 0
  
  try:
    opts, args = getopt.getopt(sys.argv[1:], tabularNavigationExp.flags)
  except getopt.GetoptError:
    raise Exception('Unknown flag')
  for opt, arg in opts:
    if opt == '-t':
      rockType = arg
    elif opt == '-r':
      random.seed(int(arg))
  
  Domain = TabularNavigation

  def rewardGen(rewards): 
    def rewardFunc(s, a):
      if s in rewards.keys():
        return rewards[s]
      else:
        return 0
    return rewardFunc

  rewardSet = []
  rewardCandNum = 6
  for candId in xrange(rewardCandNum):
    rocks = [(random.randint(width), width.randint(height)) for _ in xrange(rewardCandNum)]
    rewardSet.append(rewardGen(rocks))

  initialPhi = []
  for _ in xrange(rewardCandNum):
    initialPhi.append(random.random())
  initialPhi = map(lambda _: _ / sum(initialPhi), initialPhi)

  terminalReward = util.Counter()
  terminalReward[(width / 2, height / 2)] = 100

  experiment(Domain, width, height, responseTime, horizon, rewardCandNum, rocks, rewardSet, initialPhi, terminalReward)