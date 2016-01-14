import util
from tabularNavigation import TabularNavigation
from tabularNavigationExp import experiment
import numpy
from nWayCorridor import NWayCorridor

if __name__ == '__main__':
  width = 8
  height = 11
  # the time step that the agent receives the response
  responseTime = 5
  horizon = numpy.inf
  rocks = [(x, height - 1) for x in xrange(width)]

  rewardCandNum = width
  terminalReward = None
  Domain = NWayCorridor

  experiment(Domain, width, height, responseTime, horizon, rewardCandNum, rocks, terminalReward)