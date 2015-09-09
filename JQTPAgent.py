from learningAgents import ReinforcementAgent
from valueIterationAgents import ValueIterationAgent
import copy
import numpy

class JPQTAgent:
  def __init__(self, cmp, rewardSet, initialPhi, gamma=0.9):
    # underlying cmp
    self.cmp = cmp
    # set of possible reward functions
    self.rewardSet = rewardSet
    self.gamma = gamma
    # init belief on rewards in rewardSet
    self.phi = initialPhi
    
    # initialize VI agent for reward set for future use
    self.viAgentSet = []
    self.rewardSetSize = len(self.rewardSet)
    for idx in range(self.rewardSetSize):
      phi = [0] * self.rewardSetSize
      phi[idx] = 1
      self.viAgentSet.append(self.getVIAgent(phi))

  def getRewardFunc(self, phi):
    """
    return the mean reward function under the given belief
    """
    return lambda state: sum([reward(state) * p for reward, p in zip(self.rewardSet, phi)])

  def getVIAgent(self, phi):
    cmp = copy.deepcopy(self.cmp)
    cmp.getReward = self.getRewardFunc(phi)

    return ValueIterationAgent(cmp, discount=self.gamma)
  
  def getValue(self, state, phi, policy, horizon):
    sum = 0
    rewardFunc = self.getRewardFunc(phi)
    
    for t in range(horizon):
      sum += self.gamma ** t * sum([rewardFunc(s) * prob for s, prob in getMultipleTransitionDistr(cmp, state, policy, t)])
    
    return sum

  def getPossiblePhiAndProbs(self, state, query):
    actions = self.cmp.getPossibleActions()
    viAgent = self.getVIAgent(self.phi)
    # action -> qValue dict
    qValues = [viAgent.getQValue(state, action) for action in actions]

    # the set of new phis
    phis = []
    for action in actions:
      # copy current phi
      phi = self.phi[:]
      for idx in range(self.rewardSetSize):
        if self.viAgentSet[idx].getPolicy(state) != action:
          phi[idx] = 0
      # normalize phi
      phi = [x / sum(phi) for x in phi]
      phis.append(phi)
    
    # the probability of observing new phis
    probsUnnormed = [numpy.exp(qValue) for qValue in qValues]
    probs = [prob / sum(probsUnnormed) for prob in probsUnnormed]
    
    # belief -> prob dict
    return {phi: prob for phi, prob in zip(phis, probs)}

  def getQValue(self, state, policy, query):
    cost = self.cmp.cost(query)

    responseTime = self.cmp.getResponseTime(state)
    vBeforeResponse = self.getValue(state, self.phi, policy, responseTime)
    
    possiblePhis = self.getPossiblePhiAndProbs(state, query)
    possibleStatesAndProbs = getMultipleTransitionDistr(cmp, state, policy, responseTime)
    
    vAfterResponse = 0
    for fPhi, fPhiProb in possiblePhis:
      viAgent = self.getVIAgent(fPhi)
      values = lambda state: viAgent.getValue(state)
      estimatedValue = 0
      for fState, fStateProb in possibleStatesAndProbs:
        estimatedValue += values(fState) * fStateProb
      vAfterResponse += fPhiProb * estimatedValue

    vAfterResponse *= self.gamma ** responseTime

    return cost + vBeforeResponse + vAfterResponse
  
  def learn(self):
    # iterate optimize over policy and query
    #TODO
    pass

def getMultipleTransitionDistr(cmp, state, policy, time):
  """
  Multiply a transition matrix multiple times.
  Return state -> prob
  """
  v = {s: 0 for s in cmp.getStates()}
  v[state] = 1

  for t in range(time):
    vNext = {s: 0 for s in cmp.getStates()}
    for s in cmp.getStates():
      for nextS, nextProb in cmp.getTransitionStatesAndProbs(state, policy(state)):
        v[nextS] += v[s] * nextProb
    v = vNext