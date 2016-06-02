import util
from tabularNavigation import TabularNavigation
from tabularNavigationExp import experiment
import numpy
from nWayCorridor import NWayCorridor
import random

if __name__ == '__main__':
  width = 100
  height = 8
  # the time step that the agent receives the response
  responseTime = 5
  horizon = 12
  rocks = [(x, height - 1) for x in xrange(width)]

  rewardCandNum = 10
  terminalReward = None
  Domain = NWayCorridor

  initialPhi = [1.0 / rewardCandNum] * rewardCandNum
  
  rewards = []
  for candId in xrange(rewardCandNum):
    reward = util.Counter()
    for i in xrange(width):
      if random.random() < 0.1:
        reward[rocks[i]] = 1
    rewards.append(reward)

  experiment(Domain, width, height, responseTime, horizon, rewardCandNum, rocks, rewards, initialPhi, terminalReward)