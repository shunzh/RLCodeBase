from learningAgents import ReinforcementAgent
from valueIterationAgents import ValueIterationAgent
import copy
import numpy
import util
import pprint
import config
import random

class QTPAgent:
  def __init__(self, cmp, rewardSet, initialPhi, gamma=0.9,\
               queryIgnored=False, clusterDistance=0):
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
    Return a trained value iteratoin agent with given phi.
    So we can use getValue, getPolicy, getQValue, etc.
    """
    rewardFunc = self.getRewardFunc(phi)

    cmp = copy.deepcopy(self.cmp)
    cmp.getReward = rewardFunc
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
    """
    Compute the expected return given initial state, transient policy, and a query
    """
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
      
      # this is a stationary policy
      self.phiToPolicy[tuple(fPhi)] = lambda s, t, agent=viAgent: agent.getPolicy(s)

    return cost + vBeforeResponse + self.gamma ** responseTime * vAfterResponse
  
  def optimizePolicy(self, query):
    """
    Uses dynamic programming
    """
    v = util.Counter()
    possiblePhis = self.getPossiblePhiAndProbs(query)
    for fPhi, fPhiProb in possiblePhis:
      fViAgent = self.getVIAgent(fPhi)
      for state in self.cmp.getStates():
        v[state] += fViAgent.getValue(state) * fPhiProb

    if config.DEBUG:
      print query, "future v"
      pprint.pprint([(s, v[s]) for s in self.cmp.getStates()])

    cmp = copy.deepcopy(self.cmp)
    cmp.getReward = self.getRewardFunc(self.phi)
    responseTime = cmp.responseTime

    # backpropogate `responseTime` time steps
    viAgent = ValueIterationAgent(cmp, discount=self.gamma, iterations=responseTime, initValues=v)
    viAgent.learn()
    # this is a non-stationary policy
    pi = lambda s, t: viAgent.getPolicy(s, t+1) 
    
    if config.DEBUG:
      print query, "v func"
      pprint.pprint([(s, viAgent.getValue(s), viAgent.getPolicies(s)) for s in cmp.getStates()])
    return pi

  def respond(self, query, response):
    """
    The response is informed to the agent regarding a previous query
    """
    # such response was imagined by the agent before and the solution is bookkept
    pi = self.phiToPolicy[self.responseToPhi[response]]
    return pi


class JointQTPAgent(QTPAgent):
  def learn(self):
    state = self.cmp.state
    maxQValue = -float('INF')
    optQuery = None; optPi = None

    for q in self.cmp.queries:
      pi = self.optimizePolicy(q)
      qValue = self.getQValue(state, pi, q)
      print q, qValue
      if qValue > maxQValue:
        maxQValue = qValue
        optQuery = q
        optPi = pi
    
    q = optQuery
    pi = optPi

    if config.VERBOSE:
      print "optimized q", q

    if self.queryIgnored:
      # in both settings, plan on prior belief
      pi = lambda s, t: self.viAgent.getPolicy(s)

    return q, pi, self.getQValue(state, pi, q)


class IterativeQTPAgent(QTPAgent):
  def optimizeQuery(self, state, policy):
    """
    Enumerate relevant considering the states reachable by the transient policy.
    """
    if config.FILTER_QUERY:
      possibleStatesAndProbs = getMultipleTransitionDistr(self.cmp, state, policy, self.cmp.responseTime)
      # FIXME
      # assuming deterministic
      fState = possibleStatesAndProbs[0][0]

      queries = []
      for query in self.cmp.queries:
        # FIXME
        # overfit sightseeing problem
        if fState[2] == 1:
          if query[0] > fState[0] and query[2] == 1:
            queries.append(query)
        else:
          if query[0] < fState[0] and query[2] == -1:
            queries.append(query)
    else:
      queries = self.cmp.queries
    
    if config.VERBOSE:
      print "Considering queries", queries

    return max(queries, key=lambda q: self.getQValue(state, policy, q))
 
  def learnInstance(self):
    # there could be multiple initializations for AQTP
    # this is learning with one initial query
    state = self.cmp.state
    # learning with queries
    q = random.choice(self.cmp.queries) # initialize with a query
    print "init q", q
    
    # iterate optimize over policy and query
    counter = 0
    while True:
      prevQ = copy.deepcopy(q)

      pi = self.optimizePolicy(q)
      q  = self.optimizeQuery(state, pi)
      if config.VERBOSE:
        print "Iteration #", counter
        print "optimized q", q
      
      if q == prevQ:
        # converged
        break
      counter += 1
    
    if self.queryIgnored:
      # in both settings, plan on prior belief
      pi = lambda s: self.viAgent.getPolicy(s)
    
    return q, pi, self.getQValue(state, pi, q)
  
  def learn(self):
    results = []
    for _ in xrange(config.AQTP_RESTARTS + 1):
      results.append(self.learnInstance())
    
    # return the results with maximum q value
    return max(results, key=lambda x: x[2])

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