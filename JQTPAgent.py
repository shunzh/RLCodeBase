from learningAgents import ReinforcementAgent
from valueIterationAgents import ValueIterationAgent
import copy
import numpy
import util

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
    actions = self.cmp.getPossibleActions(state)
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

  def getQValue(self, state, policy, query, responseTime):
    cost = self.cmp.cost(query)

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
  
  def optimizeQuery(self, state, policy):
    """
    Enumerate all possible queries
    """
    responseTime = self.cmp.getResponseTime()
    return max(self.cmp.querySet, key=lambda q: self.getQValue(state, policy, q, responseTime))
  
  def optimizePolicy(self, state, query):
    """
    Uses dynamic programming
    """
    v = util.Counter()
    for state in self.cmp.getStates():
      v[state] = self.getQValue(state, None, query, 0)
      
    cmp = copy.deepcopy(self.cmp)
    cmp.getReward = self.getRewardFunc(self.phi)
    responseTime = self.cmp.getResponseTime()
    viAgent = ValueIterationAgent(cmp, iterations=responseTime, initValues=v)
    return lambda state: viAgent.getPolicy(state)

  def learn(self):
    state = self.cmp.state
    q = self.cmp.queries[0] # get a random query
    pi = None # ok to start with nothing

    # iterate optimize over policy and query
    while True:
      self.optimizePolicy(state, q)
      self.optimizeQuery(state, pi)

def getMultipleTransitionDistr(cmp, state, policy, time):
  """
  Multiply a transition matrix multiple times.
  Return state -> prob
  """
  p = {s: 0 for s in cmp.getStates()}
  p[state] = 1

  for t in range(time):
    vNext = {s: 0 for s in cmp.getStates()}
    for s in cmp.getStates():
      for nextS, nextProb in cmp.getTransitionStatesAndProbs(state, policy(state)):
        p[nextS] += p[s] * nextProb
    p = vNext