import util
from tabularNavigation import TabularNavigation
from tabularNavigationExp import experiment

if __name__ == '__main__':
  width = 21
  height = 21
  # the time step that the agent receives the response
  responseTime = 10
  horizon = 40
  rocks = [(1, 0), (0, 1),
           (width - 2, 0), (width - 1, 1), \
           (0, height - 1),\
           (width - 1, height - 1)]
  rewardCandNum = 6
  terminalReward = util.Counter()
  terminalReward[(width / 2, height / 2)] = 100
  Domain = TabularNavigation

  experiment(Domain, width, height, responseTime, horizon, rewardCandNum, rocks, terminalReward)