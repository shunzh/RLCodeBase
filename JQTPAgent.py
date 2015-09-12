from learningAgents import ReinforcementAgent
from valueIterationAgents import ValueIterationAgent
import copy
import numpy
import util

class JQTPAgent:
  def __init__(self, cmp, rewardSet, initialPhi, gamma=0.9):
    # underlying cmp
    self.cmp = cmp
    # set of possible reward functions
    self.rewardSet = rewardSet
    self.gamma = gamma
    # init belief on rewards in rewardSet
    self.phi = initialPhi
    
    # bookkeep our in-mind planning
    self.responseToPhi = util.Counter()
    self.phiToPolicy = util.Counter()

    # initialize VI agent for reward set for future use
    self.viAgentSet = []
    self.rewardSetSize = len(self.rewardSet)
    for idx in range(self.rewardSetSize):
      phi = [0] * self.rewardSetSize
      phi[idx] = 1
      
      # a VI agent based on one reward
      viAgent = self.getVIAgent(phi)
      self.viAgentSet.append(viAgent)

      print viAgent.values

  def getRewardFunc(self, phi):
    """
    return the mean reward function under the given belief
    """
    return lambda state: sum([reward(state) * p for reward, p in zip(self.rewardSet, phi)])

  def getVIAgent(self, phi):
    """
    Return a trained value iteratoin agent with given phi.
    So we can use getValue, getPolicy, getQValue, etc.
    """
    cmp = copy.deepcopy(self.cmp)
    cmp.getReward = self.getRewardFunc(phi)

    return ValueIterationAgent(cmp, discount=self.gamma)
  
  def getValue(self, state, phi, policy, horizon):
    """
    Accumulated rewards by following a fixed policy to a time horizon.
    No learning here. 
    """
    v = 0
    rewardFunc = self.getRewardFunc(phi)
    
    for t in range(1, horizon + 1):
      possibleStatesAndProbs = getMultipleTransitionDistr(self.cmp, state, policy, t)
      v += self.gamma ** t * sum([rewardFunc(s) * prob for s, prob in possibleStatesAndProbs])
    
    return v

  def getPossiblePhiAndProbs(self, query):
    actions = self.cmp.getPossibleActions(query)
    # belief -> prob dict
    distr = util.Counter()

    # consider all possible actions (responses), find out their probabilities,
    # and compute the probability of observing the next phi
    for action in actions:
      # the probability of observing this action
      actProb = 0
      phi = self.phi[:]
      for idx in range(self.rewardSetSize):
        if self.viAgentSet[idx].getPolicy(query) == action:
          actProb += phi[idx]

      # given that this response is observed, compute the next phi
      for idx in range(self.rewardSetSize):
        if self.viAgentSet[idx].getPolicy(query) != action:
          phi[idx] = 0
      # normalize phi, only record possible phis
      if sum(phi) != 0:
        phi = [x / sum(phi) for x in phi]
        distr[tuple(phi)] = actProb
        self.responseToPhi[action] = tuple(phi)
    
    return distr.items()

  def getQValue(self, state, policy, query):
    cost = self.cmp.cost(query)

    responseTime = self.cmp.responseTime
    vBeforeResponse = self.getValue(state, self.phi, policy, responseTime)
    
    possiblePhis = self.getPossiblePhiAndProbs(query)
    possibleStatesAndProbs = getMultipleTransitionDistr(self.cmp, state, policy, responseTime)
    
    vAfterResponse = 0
    for fPhi, fPhiProb in possiblePhis:
      viAgent = self.getVIAgent(fPhi)
      values = lambda state: viAgent.getValue(state)
      estimatedValue = 0
      for fState, fStateProb in possibleStatesAndProbs:
        estimatedValue += values(fState) * fStateProb
      vAfterResponse += fPhiProb * estimatedValue
      
      self.phiToPolicy[tuple(fPhi)] = lambda s, agent=viAgent: agent.getPolicy(s)

    return cost + vBeforeResponse + self.gamma ** responseTime * vAfterResponse
  
  def optimizeQuery(self, state, policy):
    """
    Enumerate all possible queries
    """
    return max(self.cmp.queries, key=lambda q: self.getQValue(state, policy, q))
  
  def optimizePolicy(self, state, query):
    """
    Uses dynamic programming
    """
    v = util.Counter()
    possiblePhis = self.getPossiblePhiAndProbs(query)
    for state in self.cmp.getStates():
      for fPhi, fPhiProb in possiblePhis:
        viAgent = self.getVIAgent(fPhi)
        values = lambda state: viAgent.getValue(state)
        v[state] += values(state) * fPhiProb
      
    cmp = copy.deepcopy(self.cmp)
    cmp.getReward = self.getRewardFunc(self.phi)
    responseTime = cmp.responseTime
    viAgent = ValueIterationAgent(cmp, iterations=responseTime, initValues=v)
    return lambda state: viAgent.getPolicy(state)

  def respond(self, query, response):
    """
    The response is informed to the agent regarding a previous query
    """
    # such response was imagined by the agent before and the solution is bookkept
    pi = self.phiToPolicy[self.responseToPhi[response]]
    return pi

  def learn(self):
    state = self.cmp.state
    q = self.cmp.queries[0] # get a random query
    pi = lambda state: self.cmp.getPossibleActions(state)[-1] # start with an arbitrary policy
    
    # iterate optimize over policy and query
    for _ in range(100):
      prevQ = copy.deepcopy(q)

      pi = self.optimizePolicy(state, q)
      q  = self.optimizeQuery(state, pi)
      print "Iteration #", _
      print "optimized pi", [(s, pi(s)) for s in self.cmp.getStates()]
      print "optimized q", q
      
      if q == prevQ:
        # converged
        return q, pi

def getMultipleTransitionDistr(cmp, state, policy, time):
  """
  Multiply a transition matrix multiple times.
  Return state -> prob
  """
  p = {s: 0 for s in cmp.getStates()}
  p[state] = 1.0
  
  for t in range(time):
    pNext = {s: 0 for s in cmp.getStates()}
    for s in cmp.getStates():
      if p[s] > 0:
        for nextS, nextProb in cmp.getTransitionStatesAndProbs(s, policy(s)):
          pNext[nextS] += p[s] * nextProb
    p = pNext.copy()
  
  assert sum(p.values()) > .99
  return p.items()