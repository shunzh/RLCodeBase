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
    rewards[0][rocks[0]] = 2
    rewards[0][rocks[1]] = -1

    rewards[1][rocks[0]] = -1
    rewards[1][rocks[1]] = 2
  elif rockType == 'default':
    pass
  else:
    raise Exception('Unknown rock type')

  terminalReward = util.Counter()
  terminalReward[(width / 2, height / 2)] = 100

  experiment(Domain, width, height, responseTime, horizon, rewardCandNum, rocks, rewards, terminalReward)