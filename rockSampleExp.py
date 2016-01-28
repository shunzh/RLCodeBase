import util
from tabularNavigation import TabularNavigation, TabularNavigationMaze
from tabularNavigationExp import experiment
import getopt
import sys
import tabularNavigationExp
import random

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
    elif opt == '-r':
      random.seed(int(arg))

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

  initialPhi = []
  for _ in xrange(rewardCandNum):
    initialPhi.append(random.random())
  initialPhi = map(lambda _: _ / sum(initialPhi), initialPhi)

  if rockType == 'corner':
    rocks = [(0, 0), (1, 0), (2, 0),\
             (0, 1), (1, 1),\
             (0, 2)]
  elif rockType == 'split':
    Domain = TabularNavigationMaze

    initialPhi = [1, 1]
    for _ in xrange(rewardCandNum - 2):
      initialPhi.append(random.random())
    initialPhi = map(lambda _: _ / sum(initialPhi), initialPhi)

    rewards[0][rocks[0]] = 1.5
    rewards[0][rocks[1]] = -1.5

    rewards[1][rocks[0]] = -1.5
    rewards[1][rocks[1]] = 1.5
  elif rockType == 'default':
    pass
  else:
    raise Exception('Unknown rock type')

  """
  for reward in rewards:
    for rock in rocks:
      print reward[rock],
    print
  """
  terminalReward = util.Counter()
  terminalReward[(width / 2, height / 2)] = 100

  experiment(Domain, width, height, responseTime, horizon, rewardCandNum, rocks, rewards, initialPhi, terminalReward)