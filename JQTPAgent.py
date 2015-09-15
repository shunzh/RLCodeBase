from learningAgents import ReinforcementAgent
from valueIterationAgents import ValueIterationAgent
import copy
import numpy
import util
import pprint

class JQTPAgent(ValueIterationAgent):
  def __init__(self, cmp, rewardSet, initialPhi, queryEnabled=True, gamma=0.9):
    # underlying cmp
    self.cmp = cmp
    # set of possible reward functions
    self.rewardSet = rewardSet
    self.gamma = gamma
    # init belief on rewards in rewardSet
    self.phi = initialPhi
    self.queryEnabled = queryEnabled
    
    if self.queryEnabled:
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
    else:
      # without query, reduce to a VI problem
      cmp.getReward = self.getRewardFunc(self.phi)
      ValueIterationAgent.__init__(self, cmp, gamma)

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

    vi = ValueIterationAgent(cmp, discount=self.gamma)
    vi.learn()
    return vi
  
  def getValue(self, state, phi, policy, horizon):
    """
    Accumulated rewards by following a fixed policy to a time horizon.
    No learning here. 
    """
    v = 0
    rewardFunc = self.getRewardFunc(phi)
    
    for t in range(horizon):
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
        if action in self.viAgentSet[idx].getPolicies(query):
          actProb += phi[idx]

      # given that this response is observed, compute the next phi
      for idx in range(self.rewardSetSize):
        if not action in self.viAgentSet[idx].getPolicies(query):
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
    print vBeforeResponse, vAfterResponse, cost + vBeforeResponse + self.gamma ** responseTime * vAfterResponse

    return cost + vBeforeResponse + self.gamma ** responseTime * vAfterResponse
  
  def optimizeQuery(self, state, policy):
    """
    Enumerate all possible queries
    """
    return max(self.cmp.queries, key=lambda q: self.getQValue(state, policy, q))
  
  def optimizePolicy(self, query):
    """
    Uses dynamic programming
    """
    v = util.Counter()
    possiblePhis = self.getPossiblePhiAndProbs(query)
    for fPhi, fPhiProb in possiblePhis:
      viAgent = self.getVIAgent(fPhi)
      for state in self.cmp.getStates():
        values = lambda s: viAgent.getValue(s)
        v[state] += values(state) * fPhiProb
      
    cmp = copy.deepcopy(self.cmp)

    cmp.getReward = self.getRewardFunc(self.phi)
    responseTime = cmp.responseTime
    viAgent = ValueIterationAgent(cmp, iterations=responseTime, initValues=v)
    pi = viAgent.learn()
    return pi

  def respond(self, query, response):
    """
    The response is informed to the agent regarding a previous query
    """
    # such response was imagined by the agent before and the solution is bookkept
    pi = self.phiToPolicy[self.responseToPhi[response]]
    return pi

  def learn(self):
    if self.queryEnabled:
      # learning with queries
      state = self.cmp.state
      q = self.cmp.queries[0] # get a random query
      pi = lambda state: self.cmp.getPossibleActions(state)[-1] # start with an arbitrary policy
      
      # iterate optimize over policy and query
      counter = 0
      while True:
        prevQ = copy.deepcopy(q)

        pi = self.optimizePolicy(q)
        q  = self.optimizeQuery(state, pi)
        print "Iteration #", counter
        print "optimized pi", [(s, pi(s)) for s in [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]]
        print "optimized q", q
        
        if q == prevQ:
          # converged
          return q, pi, self.getQValue(state, pi, q)
        counter += 1
    else:
      pi = ValueIterationAgent.learn(self)
      return None, pi, ValueIterationAgent.getValue(self, self.cmp.state)

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
  return filter(lambda (x, y): y>0, p.items())