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
  
  if rockType == 'crazy':
    width = 41
    height = 41
    responseTime = 20
    horizon = 45

  Domain = TabularNavigation
  rocks = [(1, 0), (0, 1),
           (width - 2, 0), (width - 1, 1), \
           (0, height - 1),\
           (width - 1, height - 1)]
    
  rewardCandNum = 6
  rewards = []
  for candId in xrange(rewardCandNum):
    reward = util.Counter()
    reward[rocks[candId]] = 1
    rewards.append(reward)

  initialPhi = []
  for _ in xrange(rewardCandNum):
    initialPhi.append(random.random())
  initialPhi = map(lambda _: _ / sum(initialPhi), initialPhi)

  if rockType == 'corner':
    rocks = [(0, 0), (1, 0), (2, 0),\
             (0, 1), (1, 1),\
             (0, 2)]
  elif rockType == 'crazy':
    rocks = []
    for x in xrange(4, width, 4):
      rocks.extend([(x, 0), (0, x), (x, height - 1), (width - 1, x)])
    
    rewardCandNum = len(rocks)
    for candId in xrange(rewardCandNum):
      reward = util.Counter()
      reward[rocks[candId]] = 1
      rewards.append(reward)

    for _ in xrange(rewardCandNum):
      initialPhi.append(random.random())
    initialPhi = map(lambda _: _ / sum(initialPhi), initialPhi)
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
  elif rockType == 'splitRoom':
    Domain = TabularNavigationMaze

    rewardCandNum = 8
    initialPhi = []
    for _ in xrange(rewardCandNum):
      initialPhi.append(random.random())
    initialPhi = map(lambda _: _ / sum(initialPhi), initialPhi)

    rewards = []
    for _ in xrange(rewardCandNum):
      reward = util.Counter()
      v1 = _ & 1
      reward[rocks[0]] = -2 + 3 * v1
      reward[rocks[1]] = 1 - 3 * v1

      v2 = (_ & 2) >> 1
      reward[rocks[2]] = -0.5 + v2
      reward[rocks[3]] = 0.5 - v2

      v3 = (_ & 4) >> 2
      reward[rocks[4]] = -0.5 + v3
      reward[rocks[5]] = 0.5 - v3
      rewards.append(reward)
  elif rockType == 'default':
    # we are all good
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