from learningAgents import ReinforcementAgent
from valueIterationAgents import ValueIterationAgent
import copy
import util
import pprint
import config
import random
from cmp import QueryType
import numpy
from copy import deepcopy

class QTPAgent:
  def __init__(self, cmp, rewardSet, initialPhi, queryType,
               gamma, queryIgnored=False, clusterDistance=0):
    """
    queryIgnored
      Query is asked, but the agent forgets such query is asked and planning using the prior belief.
      After a response is received, the agent will update its policy.
    """
    # underlying cmp
    self.cmp = cmp
    # set of possible reward functions
    self.rewardSet = rewardSet
    self.gamma = gamma
    # init belief on rewards in rewardSet
    self.phi = initialPhi
    self.queryType = queryType
    self.queryIgnored = queryIgnored

    self.clusterDistance = clusterDistance
    self.rewardClusters = util.Counter()
    
    self.preprocess()
    
  def preprocess(self):
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
    
    if self.queryIgnored:
      # plan on the mean rewards in this case
      self.viAgent = self.getVIAgent(self.phi)

  def getRewardFunc(self, phi):
    """
    return the mean reward function under the given belief
    """
    return lambda state: sum([reward(state) * p for reward, p in zip(self.rewardSet, phi)])
  
  def getRewardDistance(self, rewardFunc0, rewardFunc1):
    dist = sum([abs(rewardFunc0(s) - rewardFunc1(s)) for s in self.cmp.getStates()])
    dist /= len(self.cmp.getStates())
    return dist

  def getVIAgent(self, phi):
    """
    Return a trained value iteration agent with given phi.
    So we can use getValue, getPolicy, getQValue, etc.
    """
    rewardFunc = self.getRewardFunc(phi)
    cmp = deepcopy(self.cmp)
    cmp.getReward = rewardFunc
    vi = ValueIterationAgent(cmp, discount=self.gamma)
    vi.learn()
    return vi
  
  def getFiniteVIAgent(self, phi, horizon, terminalReward):
    rewardFunc = self.getRewardFunc(phi)
    cmp = deepcopy(self.cmp)
    cmp.getReward = rewardFunc
    vi = ValueIterationAgent(cmp, discount=self.gamma, iterations=horizon, initValues=terminalReward)
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

    if self.queryType == QueryType.POLICY:
      resSet = actions
      consistCond = lambda res, idx: res in self.viAgentSet[idx].getPolicies(query)
    elif self.queryType == QueryType.REWARD_SIGN:
      resSet = [-1, 0, 1]
      consistCond = lambda res, idx: numpy.sign(self.rewardSet[idx](query)) == res
    elif self.queryType == QueryType.REWARD:
      resSet = self.cmp.possibleRewardValues
      consistCond = lambda res, idx: self.rewardSet[idx](query) == res
    elif self.queryType == QueryType.NONE:
      resSet = [0]
      consistCond = lambda res, idx: True
    else:
      raise Exception('unknown type of query ' + self.queryType)

    # consider all possible responses, find out their probabilities,
    # and compute the probability of observing the next phi
    for res in resSet:
      # the probability of observing this res
      resProb = 0
      phi = self.phi[:]
      for idx in range(self.rewardSetSize):
        if consistCond(res, idx):
          resProb += phi[idx]
      
      # given that this response is observed, compute the next phi
      for idx in range(self.rewardSetSize):
        if not consistCond(res, idx):
          phi[idx] = 0
      
      # normalize phi, only record possible phis
      if sum(phi) != 0:
        phi = [x / sum(phi) for x in phi]
        distr[tuple(phi)] = resProb
        self.responseToPhi[(query, res)] = tuple(phi)
    
    return distr.items()

  def getQValue(self, state, policy, query):
    """
    Compute the expected return given initial state, transient policy, and a query
    """
    cost = self.cmp.cost(query)

    responseTime = self.cmp.responseTime
    horizon = self.cmp.horizon
    terminalReward = self.cmp.terminalReward
    vBeforeResponse = self.getValue(state, self.phi, policy, responseTime)
    
    possiblePhis = self.getPossiblePhiAndProbs(query)
    possibleStatesAndProbs = getMultipleTransitionDistr(self.cmp, state, policy, responseTime)
    
    vAfterResponse = 0
    for fPhi, fPhiProb in possiblePhis:
      if horizon == numpy.inf: viAgent = self.getVIAgent(fPhi)
      else: viAgent = self.getFiniteVIAgent(fPhi, horizon - responseTime, terminalReward)
      values = lambda state: viAgent.getValue(state)
      estimatedValue = 0
      for fState, fStateProb in possibleStatesAndProbs:
        estimatedValue += values(fState) * fStateProb
      vAfterResponse += fPhiProb * estimatedValue
      
      if horizon == numpy.inf: self.phiToPolicy[tuple(fPhi)] = lambda s, t, agent=viAgent: agent.getPolicy(s)
      else: self.phiToPolicy[tuple(fPhi)] = lambda s, t, agent=viAgent: agent.getPolicy(s, t - responseTime)

    return cost + vBeforeResponse + self.gamma ** responseTime * vAfterResponse
  
  def optimizePolicy(self, query):
    """
    Uses dynamic programming
    """
    v = util.Counter()
    possiblePhis = self.getPossiblePhiAndProbs(query)
    responseTime = self.cmp.responseTime
    horizon = self.cmp.horizon
    terminalReward = self.cmp.terminalReward

    for fPhi, fPhiProb in possiblePhis:
      if horizon == numpy.inf: fViAgent = self.getVIAgent(fPhi)
      else: fViAgent = self.getFiniteVIAgent(fPhi, horizon - responseTime, terminalReward)
      for state in self.cmp.getStates():
        v[state] += fViAgent.getValue(state) * fPhiProb

    # `responseTime` time steps
    viAgent = self.getFiniteVIAgent(self.phi, responseTime, v)
    # this is a non-stationary policy
    pi = lambda s, t: viAgent.getPolicy(s, t) 
    
    if config.DEBUG:
      print query, "v func"
      pprint.pprint([(s, viAgent.getValue(s), viAgent.getPolicies(s)) for s in cmp.getStates()])
    return pi

  def respond(self, query, response):
    """
    The response is informed to the agent regarding a previous query
    """
    # such response was imagined by the agent before and the solution is bookkept
    pi = self.phiToPolicy[self.responseToPhi[(query, response)]]
    return pi


class JointQTPAgent(QTPAgent):
  def learn(self):
    state = self.cmp.state
    maxQValue = -float('INF')
    optQuery = None; optPi = None

    for q in self.cmp.queries:
      pi = self.optimizePolicy(q)
      qValue = self.getQValue(state, pi, q)
      if config.VERBOSE: print q, qValue

      if qValue > maxQValue:
        maxQValue = qValue
        optQuery = q
        optPi = pi
    
    q = optQuery
    pi = optPi

    if config.VERBOSE:
      print "optimized q", q

    if self.queryIgnored:
      # forget the optimal pi, but plan on prior belief in this case
      pi = lambda s, t: self.viAgent.getPolicy(s, t)

    return q, pi, maxQValue


class AlternatingQTPAgent(QTPAgent):
  def __init__(self, cmp, rewardSet, initialPhi, queryType, relevance, gamma, restarts = 0):
    QTPAgent.__init__(self, cmp, rewardSet, initialPhi, queryType, gamma)
    self.relevance = relevance
    self.restarts = restarts

  def optimizeQuery(self, state, policy):
    """
    Enumerate relevant considering the states reachable by the transient policy.
    """
    possibleStatesAndProbs = getMultipleTransitionDistr(self.cmp, state, policy, self.cmp.responseTime)
    # FIXME
    # assuming deterministic
    fState = possibleStatesAndProbs[0][0]
    
    queries = []
    for query in self.cmp.queries:
      # FIXME
      # overfit sightseeing problem
      if self.relevance(fState, query): queries.append(query)
    
    if config.VERBOSE:
      print "Considering queries", queries

    if queries == []:
      # FIXME pick first query when no relevant queries
      return self.cmp.queries[0]
    else:
      return max(queries, key=lambda q: self.getQValue(state, policy, q))
 
  def learnInstance(self):
    # there could be multiple initializations for AQTP
    # this is learning with one initial query
    state = self.cmp.state
    # learning with queries
    q = random.choice(self.cmp.queries) # initialize with a query
    if config.VERBOSE: print "init q", q
    
    # iterate optimize over policy and query
    counter = 0
    while True:
      prevQ = copy.deepcopy(q)

      pi = self.optimizePolicy(q)
      q  = self.optimizeQuery(state, pi)
      if config.VERBOSE:
        print "Iteration #", counter
        print "optimized q ", q
      
      if q == prevQ:
        # converged
        break
      counter += 1
    
    if self.queryIgnored:
      # in both settings, plan on prior belief
      pi = lambda s, t: self.viAgent.getPolicy(s, t)
    
    return q, pi, self.getQValue(state, pi, q)
  
  def learn(self):
    results = []
    for _ in xrange(self.restarts + 1):
      results.append(self.learnInstance())
    
    # return the results with maximum q value
    return max(results, key=lambda x: x[2])


class RandomQueryAgent(QTPAgent):
  def learn(self):
    state = self.cmp.state
    q = random.choice(self.cmp.queries)
    pi = self.optimizePolicy(q)
    qValue = self.getQValue(state, pi, q)
    return q, pi, qValue


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
        for nextS, nextProb in cmp.getTransitionStatesAndProbs(s, policy(s, t)):
          pNext[nextS] += p[s] * nextProb
    p = pNext.copy()
  
  assert sum(p.values()) > .99
  return filter(lambda (x, y): y>0, p.items())