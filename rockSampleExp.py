import util
from tabularNavigation import TabularNavigation, TabularNavigationMaze
from tabularNavigationExp import experiment
import getopt
import sys
import tabularNavigationExp

if __name__ == '__main__':
  width = 21
  height = 21
  # the time step that the agent receives the response
  responseTime = 10
  horizon = 40
  rockType = 'default'
  
  try:
    opts, args = getopt.getopt(sys.argv[1:], tabularNavigationExp.flags)
  except getopt.GetoptError:
    raise Exception('Unknown flag')
  for opt, arg in opts:
    if opt == '-t':
      rockType = arg

  Domain = TabularNavigation
  rocks = [(1, 0), (0, 1),
           (width - 2, 0), (width - 1, 1), \
           (0, height - 1),\
           (width - 1, height - 1)]
    
  rewardCandNum = 6
  rewards = []
  for _ in xrange(rewardCandNum):
    reward = util.Counter()
    reward[rocks[_]] = 1
    rewards.append(reward)

  if rockType == 'corner':
    rocks = [(0, 0), (1, 0), (2, 0),\
             (0, 1), (1, 1),\
             (0, 2)]
  elif rockType == 'split':
    Domain = TabularNavigationMaze
    rewardCandNum = 8
    rewards = []
    for i in xrange(rewardCandNum):
      reward = util.Counter()
      v1 = i & 1
      reward[rocks[0]] = -5 + 10 * v1
      reward[rocks[1]] = 5 - 10 * v1

      v2 = (i & 2) >> 1
      print v2
      reward[rocks[2]] = -0.5 + v2
      reward[rocks[3]] = 0.5 - v2

      v3 = (i & 4) >> 2
      reward[rocks[4]] = -0.5 + v3
      reward[rocks[5]] = 0.5 - v3
      rewards.append(reward)
    for reward in rewards:
      for rock in rocks:
        print reward[rock],
      print
  elif rockType == 'default':
    pass
  else:
    raise Exception('Unknown rock type')

  terminalReward = util.Counter()
  terminalReward[(width / 2, height / 2)] = 100

  experiment(Domain, width, height, responseTime, horizon, rewardCandNum, rocks, rewards, terminalReward)