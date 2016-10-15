import util
from tabularNavigationExp import experiment
import getopt
import sys
import tabularNavigationExp
import random
import config
import numpy
from tabularNavigation import Driving

if __name__ == '__main__':
  width = 5
  height = 20
  # the time step that the agent receives the response
  horizon = height + 1
  responseTime = 0
  rewardCandNum = 3

  try:
    opts, args = getopt.getopt(sys.argv[1:], tabularNavigationExp.flags)
  except getopt.GetoptError:
    raise Exception('Unknown flag')
  for opt, arg in opts:
    if opt == '-t':
      config.para = int(arg)
    elif opt == '-n':
      config.INIT_STATE_DISTANCE = int(arg)
    elif opt == '-m':
      config.NUMBER_OF_RESPONSES = int(arg)
    elif opt == '-r':
      random.seed(int(arg))
      numpy.random.seed(int(arg))
  config.opts = '_'.join(map(str, []))
  
  def rewardGen(rewards, numerical): 
    def rewardFunc(s, a):
      if (s, a) in rewards:
        return numerical
      else:
        return 0
    return rewardFunc

  terminalReward = util.Counter()

  cmp = Driving(height, responseTime, width, height, horizon, terminalReward)
  print 'cars', cmp.cars

  ops = []
  ops.append(lambda l, car: l.update({car: 1})) # nasty driver
  ops.append(lambda l, (carX, carY): l.update({(carX, carY - 1): 1, (carX, carY): -1})) # threatening driver
  ops.append(lambda l, car: l.update({car: -1})) # nice driver
 
  rewards = []
  for op in ops:
    rewards.append(util.Counter())
    reward = rewards[-1]
    for car in cmp.cars: op(reward, car)

  rewardSet = []
  rewardSet.append(lambda s, a: rewards[0][s])
  rewardSet.append(lambda s, a: rewards[1][s])
  rewardSet.append(lambda s, a: rewards[2][s])

  initialPhi = [1.0 / rewardCandNum] * rewardCandNum
  experiment(cmp, rewardSet, initialPhi)
