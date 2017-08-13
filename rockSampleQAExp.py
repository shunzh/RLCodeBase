import util
from tabularNavigation import TabularNavigationToy
import augmentedCMP.AugmentedCMP
from cmp import QueryType
import numpy as np
from valueIterationAgents import ValueIterationAgent
import random
import time

if __name__ == '__main__':
  width = 7
  height = 7 
  # the time step that the agent receives the response
  queryType = QueryType.REWARD_SIGN
  gamma = 0.9
  
  rocks = [(1, 0), (0, 1),
           (width - 2, 0), (width - 1, 1)]
  queries = rocks
  rewardCandNum = 4
  rewards = []

  reward = util.Counter()
  reward[rocks[0]] = 1
  reward[rocks[2]] = 1
  rewards.append(reward)

  reward = util.Counter()
  reward[rocks[0]] = 1
  reward[rocks[3]] = 1
  rewards.append(reward)

  reward = util.Counter()
  reward[rocks[1]] = 1
  reward[rocks[2]] = 1
  rewards.append(reward)

  reward = util.Counter()
  reward[rocks[1]] = 1
  reward[rocks[3]] = 1
  rewards.append(reward)

  def rewardGen(reward): 
    def rewardFunc(s):
      if s in reward.keys():
        return reward[s]
      else:
        return 0
    return rewardFunc

  rewardSet = [rewardGen(reward) for reward in rewards]

  trueReward = rewardSet[0]

  initialPsi = [1.0] * len(rocks)
  initialPsi = map(lambda _: _ / sum(initialPsi), initialPsi)

  startTime = time.time()
  cmpDomain = TabularNavigationToy(queries, trueReward, gamma, width, height, np.inf, rocks, 0.5)
  domain = augmentedCMP(cmpDomain, rewardSet, initialPsi, queryType, gamma, 1, awina=True)
  initValues = domain.getVIInitial()
  
  agent = ValueIterationAgent(domain, discount=gamma, initValues=initValues)
  agent.learn()
  print time.time() - startTime
  state = domain.state
  #print 's', state
  while True:
    if domain.isTerminal(state):
      break
    
    action = agent.getPolicy(state)
    state, reward = domain.doAction(action)
    #print domain.timer, action, 's', state, 'r', reward
    #if action[0] == 'q': print domain.timer
    domain.timeElapse()