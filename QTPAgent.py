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
import scipy.stats
import operator

class QTPAgent:
  def __init__(self, cmp, rewardSet, initialPhi, queryType,
               gamma, clusterDistance=0):
    # underlying cmp
    self.cmp = cmp
    # set of possible reward functions
    self.rewardSet = rewardSet
    self.gamma = gamma
    # init belief on rewards in rewardSet
    self.phi = initialPhi
    self.queryType = queryType

    self.clusterDistance = clusterDistance
    self.rewardClusters = util.Counter()
    
    self.preprocess()
    
  def preprocess(self):
    # bookkeep our in-mind planning
    self.responseToPhi = util.Counter()
    self.phiToPolicy = util.Counter()
    horizon = self.cmp.horizon
    terminalReward = self.cmp.terminalReward

    # initialize VI agent for reward set for future use
    self.viAgentSet = util.Counter()
    self.rewardSetSize = len(self.rewardSet)

    for idx in range(self.rewardSetSize):
      phi = [0] * self.rewardSetSize
      phi[idx] = 1
      
      if horizon == numpy.inf: self.viAgentSet[idx] = self.getVIAgent(phi)
      else: self.viAgentSet[idx] = self.getFiniteVIAgent(phi, horizon, terminalReward)

  def getRewardFunc(self, phi):
    """
    return the mean reward function under the given belief
    """
    return lambda state: sum([reward(state) * p for reward, p in zip(self.rewardSet, phi)])
  
  def getVIAgent(self, phi, posterior=False):
    """
    Return a trained value iteration agent with given phi.
    So we can use getValue, getPolicy, getQValue, etc.
    """
    if posterior and tuple(phi) in self.viAgentSet.keys():
      return self.viAgentSet[tuple(phi)]
    else:
      rewardFunc = self.getRewardFunc(phi)
      cmp = deepcopy(self.cmp)
      cmp.getReward = rewardFunc
      vi = ValueIterationAgent(cmp, discount=self.gamma)
      vi.learn()
      if posterior: self.viAgentSet[tuple(phi)] = vi # bookkeep
      return vi
  
  def getFiniteVIAgent(self, phi, horizon, terminalReward, posterior=False):
    if posterior and tuple(phi) in self.viAgentSet.keys():
      return self.viAgentSet[tuple(phi)]
    else:
      rewardFunc = self.getRewardFunc(phi)
      cmp = deepcopy(self.cmp)
      cmp.getReward = rewardFunc
      vi = ValueIterationAgent(cmp, discount=self.gamma, iterations=horizon, initValues=terminalReward)
      vi.learn()
      if posterior: self.viAgentSet[tuple(phi)] = vi # bookkeep
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
    responseTime = self.cmp.getResponseTime()
    # belief -> prob dict
    distr = util.Counter()

    if self.queryType == QueryType.POLICY:
      resSet = actions
      consistCond = lambda res, idx: res in self.viAgentSet[idx].getPolicies(query, responseTime)
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
    
    return map(lambda l: (l[0], l[1]/sum(distr.values())), distr.items())

  def getQValue(self, state, policy, query):
    """
    Compute the expected return given initial state, transient policy, and a query
    """
    cost = self.cmp.cost(query)

    responseTime = self.cmp.getResponseTime()
    horizon = self.cmp.horizon
    terminalReward = self.cmp.terminalReward
    vBeforeResponse = self.getValue(state, self.phi, policy, responseTime)
    
    possiblePhis = self.getPossiblePhiAndProbs(query)
    possibleStatesAndProbs = getMultipleTransitionDistr(self.cmp, state, policy, responseTime)
    
    vAfterResponse = 0
    for fPhi, fPhiProb in possiblePhis:
      if horizon == numpy.inf: viAgent = self.getVIAgent(fPhi)
      else: viAgent = self.getFiniteVIAgent(fPhi, horizon - responseTime, terminalReward, posterior=True)
      values = lambda state: viAgent.getValue(state)
      estimatedValue = 0
      for fState, fStateProb in possibleStatesAndProbs:
        estimatedValue += values(fState) * fStateProb
      vAfterResponse += fPhiProb * estimatedValue

    return cost + vBeforeResponse + self.gamma ** responseTime * vAfterResponse
  
  def optimizePolicy(self, query):
    """
    Uses dynamic programming
    """
    v = util.Counter()
    possiblePhis = self.getPossiblePhiAndProbs(query)
    responseTime = self.cmp.getResponseTime()
    horizon = self.cmp.horizon
    terminalReward = self.cmp.terminalReward

    for fPhi, fPhiProb in possiblePhis:
      if horizon == numpy.inf: fViAgent = self.getVIAgent(fPhi)
      else: fViAgent = self.getFiniteVIAgent(fPhi, horizon - responseTime, terminalReward, posterior=True)
      for state in self.cmp.getStates():
        v[state] += fViAgent.getValue(state) * fPhiProb
      
      if horizon == numpy.inf: self.phiToPolicy[tuple(fPhi)] = lambda s, t, agent=fViAgent: agent.getPolicy(s)
      else: self.phiToPolicy[tuple(fPhi)] = lambda s, t, agent=fViAgent: agent.getPolicy(s, t - responseTime)

    # `responseTime` time steps
    viAgent = self.getFiniteVIAgent(self.phi, responseTime, v)
    # this is a non-stationary policy
    pi = lambda s, t: viAgent.getPolicy(s, t)
    
    if config.DEBUG:
      print query, "v func"
      pprint.pprint([(s, viAgent.getValue(s), viAgent.getPolicies(s)) for s in cmp.getStates()])
      
    return pi, viAgent.getValue(self.cmp.state, 0)

  def optimizeQuery(self, state, policy):
    """
    Enumerate relevant considering the states reachable by the transient policy.
    """
    responseTime = self.cmp.getResponseTime()
    horizon = self.cmp.horizon
    possibleRewardLocs = self.cmp.possibleRewardLocations
    possibleStatesAndProbs = getMultipleTransitionDistr(self.cmp, state, policy, responseTime)
    # FIXME
    # assuming deterministic
    fState = possibleStatesAndProbs[0][0]

    # FIXME
    # overfit finite horizon for simplicity
    reachableSet = filter(lambda loc: self.cmp.measure(loc, fState) < horizon - responseTime, possibleRewardLocs)
    
    queries = []
    if self.relevance == None:
      rewardFunc = self.getRewardFunc(self.phi)
      rewardVec = [rewardFunc(loc) for loc in reachableSet]

      # use the default method for query filtering
      for query in self.cmp.queries:
        possiblePhis = self.getPossiblePhiAndProbs(query)
        infGain = False
        for fPhi, fPhiProb in possiblePhis:
          rewardFunc = self.getRewardFunc(fPhi)
          postRewardVec = [rewardFunc(loc) for loc in reachableSet]
          if any(abs(v) > 1e-3 for v in postRewardVec) and scipy.stats.entropy(rewardVec, postRewardVec) > 1e-3:
            infGain = True
            break
        if infGain: queries.append(query)
    elif self.relevance == 'ipp':
      # FIXME OVERFIT
      reachableSet = filter(lambda loc: self.cmp.measure(loc, fState) + self.cmp.measure(loc, (self.cmp.width / 2, self.cmp.height / 2))\
                                        <= horizon - responseTime, possibleRewardLocs)
      rewardFunc = self.getRewardFunc(self.phi)
      rewardVec = [rewardFunc(loc) for loc in reachableSet]

      # use the default method for query filtering
      for query in self.cmp.queries:
        possiblePhis = self.getPossiblePhiAndProbs(query)
        infGain = False
        for fPhi, fPhiProb in possiblePhis:
          rewardFunc = self.getRewardFunc(fPhi)
          postRewardVec = [rewardFunc(loc) for loc in reachableSet]
          if any(abs(v) > 1e-3 for v in postRewardVec) and rewardVec.index(max(rewardVec)) != postRewardVec.index(max(postRewardVec)):
            infGain = True
            break
        if infGain: queries.append(query)
    else:
      for query in self.cmp.queries:
        if self.relevance(fState, query): queries.append(query)
    
    if config.VERBOSE:
      print "Considering queries", queries

    if config.PRINT == 'queries': print len(queries)

    if queries == []:
      # FIXME pick first query when no relevant queries
      return self.cmp.queries[-1]
    else:
      return max(queries, key=lambda q: self.getQValue(state, policy, q))
 
  def respond(self, query, response):
    """
    The response is informed to the agent regarding a previous query
    """
    # such response was imagined by the agent before and the solution is bookkept
    pi = self.phiToPolicy[self.responseToPhi[(query, response)]]
    return pi


class JointQTPAgent(QTPAgent):
  def learn(self):
    qList = []

    for q in self.cmp.queries:
      pi, qValue = self.optimizePolicy(q)
      if config.PRINT == 'qs': print qValue
      if config.VERBOSE: print q, qValue
      qList.append((q, pi, qValue))
    
    maxQValue = max(map(lambda _:_[2], qList))
    qList = filter(lambda _: _[2] == maxQValue, qList)

    return random.choice(qList)


class HeuristicAgent(QTPAgent):
  def __init__(self, cmp, rewardSet, initialPhi, queryType,
               gamma, clusterDistance=0):
    QTPAgent.__init__(self, cmp, rewardSet, initialPhi, queryType, gamma, clusterDistance)

    self.meanReward = self.getRewardFunc(self.phi)
    (self.xmin, self.xmax) = self.cmp.getReachability()
    
    self.m = config.para

  def learn(self):
    values = []

    for q in self.cmp.queries:
      possiblePhis = self.getPossiblePhiAndProbs(q)
      v = util.Counter()
      for fPhi, fPhiProb in possiblePhis:
        rewardFunc = self.getRewardFunc(fPhi)
        for s in self.cmp.getStates():
          v[q, s] += fPhiProb * (max(self.xmax[s] * rewardFunc(s), self.xmin[s] * rewardFunc(s)) - self.meanReward(s))
      v = sorted(v.items(), key=operator.itemgetter(1), reverse=True)
      values += v[:self.m]
    
    values = sorted(values, key=operator.itemgetter(1), reverse=True)
    qList = []
    sList = []
    for item in values:
      (q, s) = item[0]
      if not s in sList and not q in map(lambda _: _[0], qList):
        pi, qValue = self.optimizePolicy(q)
        qList.append((q, pi, qValue))
        sList.append(s)
        if len(qList) >= self.m: break

    maxQValue = max(map(lambda _:_[2], qList))
    qList = filter(lambda _: _[2] == maxQValue, qList)

    return random.choice(qList)


class ActiveSamplingAgent(HeuristicAgent):
  def __init__(self, cmp, rewardSet, initialPhi, queryType,
               gamma, clusterDistance=0):
    QTPAgent.__init__(self, cmp, rewardSet, initialPhi, queryType, gamma, clusterDistance)

    self.m = config.para

  def learn(self):
    hList = []

    # must be action queries
    for s in self.cmp.getStates():
      hValue = 0
      for a in self.cmp.getPossibleActions(self.cmp.state):
        bins = [0] * 10
        for idx in range(self.rewardSetSize):
          policies = self.viAgentSet[idx].getPolicies(s)
          if a in policies:
            id = min(int(10 / len(policies)), 9)
            bins[id] += 1
          else: bins[0] += 1
        hValue += scipy.stats.entropy(bins)
        
      hList.append((s, hValue))

    hList = sorted(hList, reverse=True, key=lambda _: _[1])
    hList = hList[:self.m]
    
    qList = []
    for q, h in hList:
      pi, qValue = self.optimizePolicy(q)
      qList.append((q, pi, qValue))

    maxQValue = max(map(lambda _:_[2], qList))
    qList = filter(lambda _: _[2] == maxQValue, qList)

    return random.choice(qList)


class AlternatingQTPAgent(QTPAgent):
  def __init__(self, cmp, rewardSet, initialPhi, queryType, gamma, relevance = None, restarts = 0):
    QTPAgent.__init__(self, cmp, rewardSet, initialPhi, queryType, gamma)
    self.relevance = relevance
    self.restarts = restarts

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

      pi, qValue = self.optimizePolicy(q)
      q  = self.optimizeQuery(state, pi)
      if config.VERBOSE:
        print "Iteration #", counter
        print "optimized q ", q
      
      if q == prevQ:
        # converged
        break
      counter += 1
    
    return q, pi, qValue
  
  def learn(self):
    results = []
    for _ in xrange(self.restarts + 1):
      results.append(self.learnInstance())
    
    # return the results with maximum q value
    return max(results, key=lambda x: x[2])


class RandomQueryAgent(QTPAgent):
  def learn(self):
    q = random.choice(self.cmp.queries)
    pi, qValue = self.optimizePolicy(q)
    return q, pi, qValue


class PriorTPAgent(QTPAgent):
  def __init__(self, cmp, rewardSet, initialPhi, queryType, gamma, relevance = None):
    QTPAgent.__init__(self, cmp, rewardSet, initialPhi, queryType, gamma)
    self.relevance = relevance
  
  def learn(self):
    # mean reward planner
    horizon = self.cmp.horizon
    responseTime = self.cmp.getResponseTime()
    if horizon == numpy.inf:
      meanViAgent = self.getVIAgent(self.phi)
    else:
      terminalReward = self.cmp.terminalReward
      meanViAgent = self.getFiniteVIAgent(self.phi, horizon, terminalReward, posterior=True)
    
    # respond with best query
    state = self.cmp.state
    pi = lambda s, t: meanViAgent.getPolicy(s, t)
    q  = self.optimizeQuery(state, pi)
    qValue = self.getQValue(state, pi, q)

    # update phi to policy 
    possiblePhis = self.getPossiblePhiAndProbs(q)
    for fPhi, fPhiProb in possiblePhis:
      if horizon == numpy.inf: fViAgent = self.getVIAgent(fPhi)
      else: fViAgent = self.getFiniteVIAgent(fPhi, horizon - responseTime, terminalReward, posterior=True)
      
      if horizon == numpy.inf: self.phiToPolicy[tuple(fPhi)] = lambda s, t, agent=fViAgent: agent.getPolicy(s)
      else: self.phiToPolicy[tuple(fPhi)] = lambda s, t, agent=fViAgent: agent.getPolicy(s, t - responseTime)

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