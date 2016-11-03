from QTPAgent import MILPAgent, QTPAgent
import scipy.stats
import random
import operator
import util
import config

class MILPActionAgent(MILPAgent):
  def learn(self):
    args, q = MILPAgent.learn(self)
    rewardCandNum = len(self.rewardSet)
    hList = []
    
    policyBins = self.computeDominatingPis(args, q)

    for s in args['S']:
      hValue = 0
      for a in args['A']:
        resProb = 0
        bins = [0] * len(q)
        for idx in xrange(rewardCandNum):
          if a in self.viAgentSet[idx].getPolicies(s):
            # increase the probability of observing this 
            resProb += self.phi[idx]
            # put opt policies into bins
            bins = [sum(_) for _ in zip(bins, policyBins[idx])]

        # possibly such action is consistent with none of the reward candidates
        if sum(bins) > 0:
          hValue += resProb * scipy.stats.entropy(bins)

      if config.VERBOSE: print hValue
      hList.append((s, hValue))

    # sort them nondecreasingly
    hList = sorted(hList, key=lambda _: _[1])
    hList = filter(lambda _: not scipy.isnan(_[1]), hList)
    hList = hList[:self.m]
    
    qList = []
    for q, h in hList:
      # FIXME ignore transient phase
      qValue = self.getQValue(self.cmp.state, None, q)
      qList.append((q, None, qValue))

    maxQValue = max(map(lambda _: _[2], qList))
    qList = filter(lambda _: _[2] == maxQValue, qList)

    return random.choice(qList)

class ActiveSamplingAgent(QTPAgent):
  def __init__(self, cmp, rewardSet, initialPhi, queryType, gamma):
    QTPAgent.__init__(self, cmp, rewardSet, initialPhi, queryType, gamma)

    self.m = 1

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
            bins[id] += self.phi[idx]
          else: bins[0] += self.phi[idx]
        #print s, a, bins
        hValue += scipy.stats.entropy(bins)
        
      hList.append((s, hValue))

    hList = sorted(hList, reverse=True, key=lambda _: _[1])
    hList = filter(lambda _: not scipy.isnan(_[1]), hList)
    #print hList
    hList = hList[:self.m]
    
    qList = []
    for q, h in hList:
      # FIXME ignore transient phase
      qValue = self.getQValue(self.cmp.state, None, q)
      qList.append((q, None, qValue))

    maxQValue = max(map(lambda _:_[2], qList))
    qList = filter(lambda _: _[2] == maxQValue, qList)

    return random.choice(qList)


class HeuristicAgent(QTPAgent):
  def __init__(self, cmp, rewardSet, initialPhi, queryType,
               gamma, clusterDistance=0):
    QTPAgent.__init__(self, cmp, rewardSet, initialPhi, queryType, gamma, clusterDistance)

    self.meanReward = self.getRewardFunc(self.phi)
    (self.xmin, self.xmax) = self.cmp.getReachability()
    
    self.m = 1

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
