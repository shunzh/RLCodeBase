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
  height = 30
  # the time step that the agent receives the response
  horizon = height + 1
  responseTime = 0
  rewardCandNum = 5

  try:
    opts, args = getopt.getopt(sys.argv[1:], tabularNavigationExp.flags)
  except getopt.GetoptError:
    raise Exception('Unknown flag')
  for opt, arg in opts:
    if opt == '-t':
      config.TRAJECTORY_LENGTH = int(arg)
    elif opt == '-n':
      config.INIT_STATE_DISTANCE = int(arg)
    elif opt == '-m':
      config.NUMBER_OF_RESPONSES = int(arg)
    elif opt == '-r':
      random.seed(int(arg))
      numpy.random.seed(int(arg))
  config.opts = '_'.join(map(str, [config.TRAJECTORY_LENGTH]))
  
  def rewardGen(rewards, numerical): 
    def rewardFunc(s, a):
      if (s, a) in rewards:
        return numerical
      else:
        return 0
    return rewardFunc

  terminalReward = util.Counter()

  cmp = Driving(5, responseTime, width, height, horizon, terminalReward)

  ops = []
  nastyDriver = lambda l, car: l.update({car: 1})
  ops.append(nastyDriver)
  dangerousDriver = lambda l, (carX, carY): l.update({(carX, carY - 1): 1, (carX, carY): -10})
  ops.append(dangerousDriver)
  niceDriver = lambda l, car: l.update({car: -10})
  ops.append(niceDriver)
 
  rewards = []
  for op in ops:
    rewards.append(util.Counter())
    reward = rewards[-1]
    for car in cmp.cars: op(reward, car)
    #print reward
    
  leftPreferred = util.Counter()
  rightPreferred = util.Counter()
  for y in xrange(height):
    leftPreferred[(0, y)] = 0.01
    rightPreferred[(width - 1, y)] = 0.01
  for car in cmp.cars: niceDriver(leftPreferred, car)
  for car in cmp.cars: niceDriver(rightPreferred, car)
  

  rewardSet = []
  rewardSet.append(lambda s, a: rewards[0][s])
  rewardSet.append(lambda s, a: rewards[1][s])
  rewardSet.append(lambda s, a: rewards[2][s])
  rewardSet.append(lambda s, a: leftPreferred[s])
  rewardSet.append(lambda s, a: rightPreferred[s])

  initialPhi = [1.0 / rewardCandNum] * rewardCandNum
  #initialPhi = [0.1, 0.3, 0.4, 0.1, 0.1]
  experiment(cmp, rewardSet, initialPhi)
