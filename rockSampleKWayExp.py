import util
from tabularNavigation import TabularNavigation, TabularNavigationToy,\
  TabularNavigationKWay
from tabularNavigationExp import experiment
import getopt
import sys
import tabularNavigationExp
import random
import config
import numpy

if __name__ == '__main__':
  width = height = 8
  # the time step that the agent receives the response
  responseTime = 0
  horizon = height + 1
  rockNum = 5
  rewardCandNum = 5

  numOfActions = config.para
  # if we are going to compare with action queries, enable the following for convenience
  # we let the number of responses be consistent with the number of actions
  #config.NUMBER_OF_RESPONSES = numOfActions
  
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
  
  Domain = TabularNavigationKWay
  
  def rewardGen(rewards, numerical): 
    def rewardFunc(s, a):
      if (s, a) in rewards:
        return numerical
      else:
        return 0
    return rewardFunc

  rewardSet = []

  rocks = [((x, y), a) for a in xrange(numOfActions)\
                       for y in xrange(height)\
                       for x in xrange(TabularNavigationKWay.getNumOfStatesPerRow(y))]
  for candId in xrange(rewardCandNum):
    sampledRocks = random.sample(rocks, rockNum)
    rewardSet.append(rewardGen(sampledRocks, random.random()))
  """
  # a case where trajectory query (actualy state-action preference query) has worse performance than policy queries
  rewardSet.append(rewardGen([((0, 0), 0), ((0, 1), 0)], 1))
  rewardSet.append(rewardGen([((0, 0), 1), ((0, 1), 0)], 1))
  rewardSet.append(rewardGen([((0, 0), 0), ((0, 1), 1)], 1))
  rewardSet.append(rewardGen([((0, 0), 1), ((0, 1), 1)], 1))
  """

  initialPhi = [1.0 / rewardCandNum] * rewardCandNum

  terminalReward = util.Counter()

  experiment(Domain, width, height, responseTime, horizon, rewardCandNum, rewardSet, initialPhi, terminalReward)
