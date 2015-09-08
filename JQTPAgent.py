from learningAgents import ReinforcementAgent
from valueIterationAgents import ValueIterationAgent
import util
import copy

class JPQTAgent:
  def __init__(self, cmp, rewardSet, initialPhi=None, gamma=0.9):
    self.cmp = cmp
    self.rewardSet = rewardSet
    self.gamma = gamma
    
    self.n = self.cmp.n
    # init belief on rewards with uniform distribution
    self.phi = initialPhi or [1.0 / self.n] * self.n

  def getRewardFunc(self, state, phi):
    #TODO

  def getReward(self, state, phi):
    """
    uses our belief
    """
    #TODO

  def getOptValues(self, phi):
    cmp = copy.deepcopy(self.cmp)
    cmp.rewardFunc = self.getRewardFunc(phi)

    viAgent = ValueIterationAgent(cmp, discount=self.gamma)
    return viAgent.getValues()
  
  def getValue(self, state, phi, pi, horizon):
    sum = 0
    
    for t in range(horizon):
      sum += self.gamma ** t * self.getReward(state, self.phi)
      state = util.chooseFromDistribution(self.cmp.getTransitionStatesAndProbs(state))
    
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