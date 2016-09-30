import util
from tabularNavigation import TabularNavigation, TabularNavigationToy,\
  TabularNavigationKWay
from tabularNavigationExp import experiment
import getopt
import sys
import tabularNavigationExp
import random
import config

if __name__ == '__main__':
  width = 1
  height = 3
  # the time step that the agent receives the response
  responseTime = 0
  horizon = height + width + 1
  rockNum = 2
  rewardCandNum = 4

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
      rewardCandNum = int(arg)
    elif opt == '-m':
      config.NUMBER_OF_RESPONSES = int(arg)
    elif opt == '-r':
      random.seed(int(arg))
  config.opts = '_'.join(map(str, [config.para, rewardCandNum]))
  
  Domain = TabularNavigationKWay
  
  def rewardGen(rewards, numerical): 
    def rewardFunc(s, a):
      if (s, a) in rewards:
        return numerical
      else:
        return 0
    return rewardFunc

  rewardSet = []

  rocks = [((x, y), a) for x in xrange(width) for y in xrange(height) for a in xrange(numOfActions)]
  for candId in xrange(rewardCandNum):
    rewardSet.append(rewardGen(random.sample(rocks, rockNum), 1.0 / rockNum))
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
