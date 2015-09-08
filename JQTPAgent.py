from learningAgents import ReinforcementAgent
from valueIterationAgents import ValueIterationAgent
import util
import copy

class JPQTAgent:
  def __init__(self, cmp, rewardSet, initialPhi, gamma=0.9):
    # underlying cmp
    self.cmp = cmp
    # set of possible reward functions
    self.rewardSet = rewardSet
    self.gamma = gamma
    # init belief on rewards in rewardSet
    self.phi = initialPhi

  def getRewardFunc(self, phi):
    """
    return the mean reward function under the given belief
    """
    return lambda state: sum([reward(state) * p for reward, p in zip(self.rewardSet, phi)])

  def getOptValues(self, phi):
    cmp = copy.deepcopy(self.cmp)
    cmp.rewardFunc = self.getRewardFunc(phi)

    viAgent = ValueIterationAgent(cmp, discount=self.gamma)
    return lambda state: viAgent.getValue(state)
  
  def getValue(self, state, phi, pi, horizon):
    sum = 0
    rewardFunc = self.getRewardFunc(phi)
    
    # only sample one trajectory
    for t in range(horizon):
      sum += self.gamma ** t * sum([rewardFunc(state) * prob for state, prob in getMultipleTransitionMatrix(cmp, policy, t)])
    
    return sum

  def getPossiblePhiAndProbs(self):
    #TODO
    pass

  def getQValue(self, state, policy, query):
    cost = self.cmp.cost(query)

    responseTime = self.cmp.getResponseTime(state)
    vBeforeResponse = self.getValue(state, self.phi, policy, responseTime)
    
    possiblePhis = self.getPossiblePhiAndProbs()
    possibleStatesAndProbs = getMultipleTransitionMatrix(cmp, policy, responseTime)
    
    vAfterResponse = 0
    for fPhi, fPhiProb in possiblePhis:
      values = self.getOptValues(fPhi)
      estimatedValue = 0
      for fState, fStateProb in possibleStatesAndProbs:
        estimatedValue += values(fState) * fStateProb
      vAfterResponse += fPhiProb * estimatedValue

    vAfterResponse *= self.gamma ** responseTime

    return cost + vBeforeResponse + vAfterResponse
  
  def learn(self):
    # iterate optimize over policy and query
    #TODO

def getTransitionMatrix(cmp, policy):
  #TODO
  pass

def getMultipleTransitionMatrix(cmp, policy, time):
  """
  Multiply a transition matrix multiple times
  """
  #TODO
  pass