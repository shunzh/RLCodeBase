import util
from tabularNavigation import TabularNavigationToy
from AugmentedCMP import AugmentedCMP
from cmp import QueryType
import numpy as np
from valueIterationAgents import ValueIterationAgent

if __name__ == '__main__':
  width = 11
  height = 11
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

  cmpDomain = TabularNavigationToy(queries, trueReward, gamma, width, height, np.inf, 0.9)
  domain = AugmentedCMP(cmpDomain, rewardSet, initialPsi, queryType, gamma, 1)
  
  agent = ValueIterationAgent(domain, discount=gamma)
  agent.learn()
  state = domain.state
  print 's', state
  while True:
    if domain.isTerminal(state):
      break
    
    action = agent.getPolicy(state)
    state, reward = domain.doAction(action)
    print action, 's', state, 'r', reward