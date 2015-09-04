from learningAgents import ReinforcementAgent
from valueIterationAgents import ValueIterationAgent
import util

class JPQTAgent:
  def __init__(self, cmp, rewardSet, gamma=0.9):
    self.cmp = cmp
    self.rewardSet = rewardSet
    self.gamma = gamma
    
    self.n = self.cmp.n
    # init belief on rewards with uniform distribution
    self.phi = [1.0 / self.n] * self.n

  def getRewardFunc(self, state, phi):
    #TODO

  def getReward(self, state, phi):
    """
    uses our belief
    """
    #TODO

  def getOptValue(self, state, phi):
    cmp = copy.deepcopy(self.cmp)
    cmp.rewardFunc = self.getRewardFunc(phi)

    viAgent = ValueIterationAgent(cmp, discount=self.gamma)
    return viAgent.getValue(state)
  
  def getValue(self, state, phi, pi, horizon):
    sum = 0
    
    for t in range(horizon):
      sum += self.gamma ** t * self.getReward(state, self.phi)
      state = util.chooseFromDistribution(self.cmp.getTransitionStatesAndProbs(state))
    
    return sum

  def getBeliefProb(self):
    pass

  def getQValue(self, state, policy, query):
    cost = self.cmp.cost(query)

    responseTime = self.cmp.getResponseTime(state)
    vBeforeResponse = self.getValue(state, self.phi, policy, responseTime)
    vAfterResponse = self.gamma ** responseTime * 
    
    return cost + vBeforeResponse
  
  def learn(self):
    # iterate optimize over policy and query
    #TODO