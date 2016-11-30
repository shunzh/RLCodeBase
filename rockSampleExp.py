import util
from tabularNavigation import RockCollection
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
  rewardCandNum = 5

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
    elif opt == '-y':
      rewardVar = int(arg)
    elif opt == '-r':
      random.seed(int(arg))
      numpy.random.seed(int(arg))
  config.opts = '_'.join(map(str, [rewardCandNum, config.NUMBER_OF_QUERIES, config.NUMBER_OF_RESPONSES, rewardVar]))
  
  def rewardGen(rewards, numerical): 
    def rewardFunc(s, a):
      if s in rewards:
        return numerical
      else:
        return 0
    return rewardFunc
  
  def reward2Gen(rewards1, rewards2, numerical1, numerical2):
    def rewardFunc(s, a):
      if s in rewards1:
        return numerical1
      elif s in rewards2:
        return numerical2
      else:
        return 0
    return rewardFunc
 
  Domain = RockCollection
  terminalReward = util.Counter()
  cmp = Domain(responseTime, width, height, horizon = horizon, terminalReward = terminalReward)

  rewardSet = []
  
 
  # THE GENERAL CASE
  # don't set rewards on terminal states (where y == height - 1)
  rocks = random.sample(filter(lambda (x, y): y < cmp.height - 1, cmp.getStates()), rockNum)

  if rewardVar == 1:
    rewardCandBonus = [1] * rewardCandNum
  elif rewardVar == 2:
    rewardCandBonus = [2] * (rewardCandNum / 2) + [1] * (rewardCandNum - rewardCandNum / 2)
  elif rewardVar == 3:
    rewardCandBonus = [5] + [1] * (rewardCandNum - 1)
  else:
    raise Exception('unknown rewardVar')

  # sample rockNum rocks from the state space
  for candId in xrange(rewardCandNum):
    # we have some more valuable rocks
    bonus = random.sample(rocks, rockNum / rewardCandNum)
    pits = [_ for _ in rocks if not _ in bonus] # pits = rocks \ bonus
    rewardSet.append(reward2Gen(bonus, pits, rewardCandBonus[candId], -1))

  initialPhi = [1.0 / rewardCandNum] * rewardCandNum

  experiment(cmp, rewardSet, initialPhi)
