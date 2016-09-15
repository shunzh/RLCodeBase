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
  width = 5
  height = 10
  # the time step that the agent receives the response
  responseTime = 0
  horizon = height + width + 1
  rockNum = 5
  rewardCandNum = 5

  # since we are going to compare with action queries,
  # we let the number of responses be consistent with the number of actions
  config.NUMBER_OF_RESPONSES = width
  
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
  
  if rockNum == 0:
    Domain = TabularNavigationToy
  else:
    #Domain = TabularNavigation
    Domain = TabularNavigationKWay
  
  def rewardGen(rewards, numerical): 
    def rewardFunc(s, a):
      if s in rewards:
        return numerical
      else:
        return 0
    return rewardFunc

  rewardSet = []
  rocks = [(x, y) for x in xrange(width) for y in xrange(height)]
  for candId in xrange(rewardCandNum):
    rewardSet.append(rewardGen(random.sample(rocks, rockNum), 1.0 / rockNum))

  initialPhi = [1.0 / rewardCandNum] * rewardCandNum

  terminalReward = util.Counter()

  experiment(Domain, width, height, responseTime, horizon, rewardCandNum, rewardSet, initialPhi, terminalReward)
