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
    print 'unknown flag encountered'
  for opt, arg in opts:
    if opt == '-t':
      rockType = arg

  Domain = TabularNavigation
  rocks = [(1, 0), (0, 1),
           (width - 2, 0), (width - 1, 1), \
           (0, height - 1),\
           (width - 1, height - 1)]
  if rockType == 'corner':
    rocks = [(0, 0), (1, 0), (2, 0),\
             (0, 1), (1, 1),\
             (0, 2)]
  elif rockType == 'split':
    Domain = TabularNavigationMaze

  rewardCandNum = 6
  terminalReward = util.Counter()
  terminalReward[(width / 2, height / 2)] = 100

  experiment(Domain, width, height, responseTime, horizon, rewardCandNum, rocks, terminalReward)