from QTPAgent import QTPAgent
import util
import easyDomains
from cmp import QueryType
import config
import numpy
import lp
import copy

class GreedyConstructionPiAgent(QTPAgent):
  def __init__(self, cmp, rewardSet, initialPhi, queryType, gamma, qi=False):
    """
    qi: query iteration
    """
    QTPAgent.__init__(self, cmp, rewardSet, initialPhi, queryType, gamma)
    # do query iteration?
    self.qi = qi
    self.m = 1
    
    if hasattr(self, 'computePiValue'):
      # policy gradient agent has different ways to compute values..
      self.computeV = lambda pi, S, A, r, horizon: self.computePiValue(pi, r, horizon)
    else:
      self.computeV = lambda pi, S, A, r, horizon: lp.computeValue(pi, r, S, A)

  def computeDominatingPis(self, args, q):
    """
    args, q: from context in learn
    policyBins[idx][i] == 1 iff i-th policy dominates reward idx
    """
    policyBins = util.Counter()
    rewardCandNum = len(args['R'])
    for rewardId in xrange(rewardCandNum):
      # the values of the policies in the query under this reward candidate
      piValues = {idx: self.computeV(q[idx], args['S'], args['A'], args['R'][rewardId], self.cmp.horizon) for idx in xrange(len(q))}
      maxValue = max(piValues.values())
      # we assumed the first consistent response is returned
      maxIdx = piValues.values().index(maxValue)
      policyBins[rewardId] = [1 if idx == maxIdx else 0 for idx in xrange(len(q))]
    return policyBins
  
  def learn(self):
    args = easyDomains.convert(self.cmp, self.rewardSet, self.phi)
    self.args = args # save a copy
    rewardCandNum = len(self.rewardSet)

    responseTime = self.cmp.getResponseTime()
    horizon = self.cmp.horizon
    terminalReward = self.cmp.terminalReward

    if self.queryType == QueryType.ACTION:
      k = len(args['A'])
    else:
      k = config.NUMBER_OF_RESPONSES

    # now q is a set of policy queries
    bestQ = None
    bestEUS = -numpy.inf
    
    q = []
    args['maxV'] = [-numpy.inf] * rewardCandNum
    # keep a copy of currently added policies. may not be used.
    # note that this is passing by inference
    args['q'] = q 
    
    # start with the prior optimal policy
    x = self.getFiniteVIAgent(self.phi, horizon, terminalReward, posterior=True).x
    # start adding following policies
    for i in range(1, k):
      if config.VERBOSE: print 'iter.', i
      x = self.findNextPolicy(**args)
      q.append(x)

      args['maxV'] = []
      for rewardId in xrange(rewardCandNum):
        args['maxV'].append(max([self.computeV(pi, args['S'], args['A'], args['R'][rewardId], horizon) for pi in q]))
      if config.VERBOSE: print 'maxV', args['maxV']

    objValue = sum(args['maxV'][idx] * self.phi[idx] for idx in range(rewardCandNum))
    if config.VERBOSE: print 'eus value', objValue

    # query iteration
    # for each x \in q, what is q -> x; \psi? replace x with the optimal posterior policy
    if self.qi:
      # FIXME need debugging
      numOfIters = 0
      while True:
        # compute dominance
        policyBins = self.computeDominatingPis(args, q)

        # one iteration
        newQ = []
        for i in range(k):
          # which reward candidates the i-th policy dominates?
          # psi is not normalized, which is fine, since we only needs the optimizing policy
          psi = [self.phi[idx] if policyBins[idx][i] == 1 else 0 for idx in xrange(rewardCandNum)]
          if config.VERBOSE: print i, psi
          agent = self.getFiniteVIAgent(psi, horizon - responseTime, terminalReward, posterior=True)
          newQ.append(agent.x)

        # compute new eus
        newObjValue = lp.computeObj(newQ, self.phi, args['S'], args['A'], args['R'])
        if config.VERBOSE: print newObjValue
        assert newObjValue >= objValue - 0.001, '%f turns to %f' % (objValue, newObjValue)
        numOfIters += 1
        if newObjValue <= objValue: break
        else:
          objValue = newObjValue
          q = newQ
      if config.VERBOSE: print numOfIters

    if self.queryType == QueryType.POLICY:
      # if asking policies directly, then return q
      #return q, objValue # THIS RETURNS EUS, NOT EPU
      return q, None
    if self.queryType == QueryType.PARTIAL_POLICY:
      idx = 0
      objValue = self.getQValue(self.cmp.state, None, q)
      qP = copy.copy(q)

      while True:
        # iterate over all the policies, remove one state pair of each
        # but make sure the EUS of the new set is unchaged
        x = qP[idx]
        xOld = x.copy()
        
        success = False
        for key in util.randomly(x.keys()):
          x.pop(key)
          print self.getQValue(self.cmp.state, None, qP), objValue 
          if self.getQValue(self.cmp.state, None, qP) == objValue:
            success = True
            break
          else:
            x = xOld.copy()
        
        if not success: break
        #print idx, len(x)
        idx = (idx + 1) % len(q)
      
      return qP
    elif self.queryType == QueryType.DEMONSTRATION:
      # if we already build a set of policies, but the query type is demonstration
      # we sample trajectories from these policies as a query
      # note that another way is implemented in MILPDemoAgent, which choose the next policy based on the demonstrated trajectories.
      qu = [self.sampleTrajectory(x) for x in q]
      return qu
    elif self.queryType in [QueryType.SIMILAR, QueryType.ACTION]:
      # implemented in a subclass, do nothing here
      pass
    else:
      raise Exception('Query type not implemented for MILP.')

    return args, q


class MILPAgent(GreedyConstructionPiAgent):
  def findNextPolicy(self, S, A, R, T, s0, psi, maxV, q):
    return lp.milp(S, A, R, T, s0, psi, maxV)


class MILPDemoAgent(MILPAgent):
  # greedily construct a set of policies for demonstration
  # assume the first i policies are demonstrated to the operator when deciding the (i+1)-st policy
  def __init__(self, cmp, rewardSet, initialPhi, queryType, gamma, heuristic=False):
    self.heuristic = heuristic
    MILPAgent.__init__(self, cmp, rewardSet, initialPhi, queryType, gamma)

  def learn(self):
    args = easyDomains.convert(self.cmp, self.rewardSet, self.phi)
    rewardCandNum = len(self.rewardSet)

    if self.queryType == QueryType.DEMONSTRATION:
      k = config.NUMBER_OF_RESPONSES
    else:
      raise Exception("query type not implemented")

    # now q is a set of TRAJECTORIES
    q = []
    for i in range(k):
      if i == 0:
        args['maxV'] = [0] * rewardCandNum
      else:
        # find the optimal policy so far that achieves the best on each reward candidate
        args['maxV'] = []
        for rewardId in xrange(rewardCandNum):
          args['maxV'].append(max([self.computeV(pi, args['S'], args['A'], args['R'][rewardId], self.cmp.horizon) for pi in q]))
      x = lp.milp(**args)
      if self.heuristic:
        #TODO what to do with this x for demonstration purpose
        pass
      q.append(self.sampleTrajectory(x))
    
    objValue = self.getQValue(self.cmp.state, None, q)

    if self.queryType == QueryType.DEMONSTRATION:
      return q, None, objValue
    else:
      raise Exception("query type not implemented")

