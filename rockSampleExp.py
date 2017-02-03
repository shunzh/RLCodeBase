import util
from tabularNavigation import RockCollection, ThreeStateToy
from tabularNavigationExp import experiment
import getopt
import sys
import tabularNavigationExp
import random
import config
import numpy

if __name__ == '__main__':
  width = 5
  height = 30
  # the time step that the agent receives the response
  responseTime = 0
  horizon = height + 1
  rewardVar = 1
  rockNum = 20
  
  try:
    opts, args = getopt.getopt(sys.argv[1:], tabularNavigationExp.flags)
  except getopt.GetoptError:
    raise Exception('Unknown flag')
  for opt, arg in opts:
    if opt == '-t':
      config.TRAJECTORY_LENGTH = int(arg)
    elif opt == '-n':
      if int(arg) == 5:
        pass
      elif int(arg) == 10:
        rockNum = 40
        rewardCandNum = 10
      else:
        raise Exception('undefined number of rocks')
    elif opt == '-k':
      config.NUMBER_OF_RESPONSES = int(arg)
    elif opt == '-r':
      random.seed(int(arg))
      numpy.random.seed(int(arg))
  
  terminalReward = util.Counter()
  cmp = RockCollection(responseTime, width, height, horizon, terminalReward, rockNum)
  #cmp = ThreeStateToy(responseTime, horizon, terminalReward)

  # rock collection
  ws = [(x, y) for x in numpy.arange(-1,1,0.2) for y in numpy.arange(-1,1,0.2)]
  # three states
  #ws = [(-1,), (0,), (1,)]
  rewardCandNum = len(ws)

  initialPhi = [1.0 / rewardCandNum] * rewardCandNum

  config.opts = '_'.join(map(str, [rewardCandNum, config.NUMBER_OF_QUERIES, config.NUMBER_OF_RESPONSES]))

  experiment(cmp, ws, initialPhi)
