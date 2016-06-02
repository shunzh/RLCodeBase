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
  responseTime = 0
  horizon = 40
  
  Domain = TabularNavigation
  rocks = []
    
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

  terminalReward = util.Counter()
  terminalReward[(width / 2, height / 2)] = 100

  experiment(Domain, width, height, responseTime, horizon, rewardCandNum, rocks, rewards, initialPhi, terminalReward)