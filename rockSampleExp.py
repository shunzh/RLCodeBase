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
  rewardCandNum = 50
  
  try:
    opts, args = getopt.getopt(sys.argv[1:], tabularNavigationExp.flags)
  except getopt.GetoptError:
    raise Exception('Unknown flag')
  for opt, arg in opts:
    if opt == '-t':
      config.TRAJECTORY_LENGTH = int(arg)
    elif opt == '-n':
      rewardCandNum = int(arg)
    elif opt == '-k':
      config.NUMBER_OF_RESPONSES = int(arg)
    elif opt == '-r':
      random.seed(int(arg))
      numpy.random.seed(int(arg))
  
  terminalReward = util.Counter()
  
  # three-state domain
  """
  cmp = ThreeStateToy(responseTime, horizon, terminalReward)
  ws = [(-1,), (1,)]
  """

  # rock collection
  cmp = RockCollection(responseTime, width, height, horizon, terminalReward, rockNum)
  ws = [(random.random(), random.random(), random.random()) for _ in range(50)]

  rewardCandNum = len(ws)

  initialPhi = [1.0 / rewardCandNum] * rewardCandNum

  config.opts = '_'.join(map(str, [rewardCandNum, config.NUMBER_OF_RESPONSES]))

  experiment(cmp, ws, initialPhi)
