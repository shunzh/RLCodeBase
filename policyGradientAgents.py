from QTPAgent import GreedyConstructionPiAgent
import numpy
import random
import config
from valueIterationAgents import ValueIterationAgent
import easyDomains
from cmp import QueryType
from copy import deepcopy

class StepSize:
  def iterate(self):
    pass
  
  def reset(self):
    pass

class DiminishingStepSize(StepSize):
  def __init__(self, alpha):
    self.initAlpha = alpha
    self.k = 0.0
  
  def getAlpha(self):
    return self.initAlpha / numpy.sqrt(self.k)
  
  def iterate(self):
    self.k += 1
  
  def reset(self):
    self.k = 0.0

class ConstantStepSize(StepSize):
  def __init__(self, alpha):
    self.alpha = alpha

  def getAlpha(self):
    return self.alpha


class PolicyGradientQueryAgent(GreedyConstructionPiAgent):
  """
  This finds the next policy by gradient descent using EUS as the objective function
  """
  def __init__(self, cmp, rewardSet, initialPhi, queryType, feat, featLength, gamma):
    self.feat = feat
    self.featLength = featLength
    self.stepSize = DiminishingStepSize(0.1)
    GreedyConstructionPiAgent.__init__(self, cmp, rewardSet, initialPhi, queryType, gamma)

  def getFiniteVIAgent(self, phi, horizon, terminalReward, posterior=False):
    if posterior and tuple(phi) in self.viAgentSet.keys():
      # bookkeep posterior optimal policies
      return self.viAgentSet[tuple(phi)]
    else:
      rewardFunc = self.getRewardFunc(phi)
      cmp = deepcopy(self.cmp)
      cmp.getReward = rewardFunc
      if posterior:
        a = PolicyGradientAgent(cmp, self.feat, self.featLength, discount=self.gamma, horizon=horizon)
      else:
        raise Exception('not implemented')
      a.learn()
      if posterior: self.viAgentSet[tuple(phi)] = a # bookkeep
      return a
 
  def thetaToOccupancy(self, theta):
    actions = self.cmp.getPossibleActions()

    def getSoftmaxActProb(s, a):
      maxV = max(numpy.dot(theta, self.feat(s, b)) for b in actions)
      actProbs = {b: numpy.exp(numpy.dot(theta, self.feat(s, b)) - maxV) for b in actions}
      return actProbs[a] / sum(actProbs.values())
    
    def getLinearActProb(s, a):
      idx = actions.index(a)
      positiveTheta = map(lambda _: abs(_), theta)
      return (positiveTheta[idx] + 0.001) / (sum(positiveTheta) + 0.001 * len(actions))

    if config.POLICY_TYPE == 'softmax':
      return getSoftmaxActProb
    elif config.POLICY_TYPE == 'linear':
      return getLinearActProb
    else: raise Exception('unknown policy type')

  def findNextPolicy(self, S, A, R, T, s0, psi, maxV):
    """
    Same arguments as lp.milp
    Return: next policy to add. It's a parameter, not occupancy
    
    FIXME not re-using the code in PolicyGradientAgent. they are very similar classes. shall we?
    """
    # start with a `trivial' controller
    horizon = self.cmp.horizon
    bestTheta = None
    bestValue = -numpy.inf
    
    #TEST
    """
    for th1 in numpy.arange(0, 10.1, 1):
      for th2 in numpy.arange(0, 10.1, 1):
        if th1 + th2 <= 10:
          theta = (th1, th2, 10 - th1 - th2)
          print self.computeObjValue(theta, psi, R, horizon, maxV),
        else:
          print 'nan',
      print
    """
    
    # compute the derivative of EUS
    for rspTime in xrange(3):
      if config.VERBOSE: print rspTime

      theta = [-0.5 + random.random() for _ in xrange(self.featLength)] # baseline
      #theta = [0] * self.featLength

      self.stepSize.reset()
      stopCounter = 0

      for iterStep in xrange(200):
        pi = self.thetaToOccupancy(theta)
        # u is a list of state action pairs
        # this is still policy query.. we sample to the task horizon
        accG = numpy.array((0.0,) * self.featLength)
        
        self.stepSize.iterate()

        # sample one trajectory here, might be used when this policy dominates in any reward candidate 
        u = self.sampleTrajectory(pi, s0, horizon, 'saPairs')
        for rIdx in range(len(R)):
          ret = self.computePiValue(pi, R[rIdx], horizon)
          #ret = sum(R[rIdx](s, a) for s, a in u)
          if config.DEBUG: print 'ret in reward id', rIdx, 'is', ret

          if ret > maxV[rIdx]:
            # here is where the non-smoothness comes from
            # only add the derivative when the accumulated return is larger than the return obtained by the
            # best policy in the query set
            futureRet = 0
            for s, a in reversed(u):
              futureRet += R[rIdx](s, a)
              
              if config.POLICY_TYPE == 'softmax':
                # softmax derivative
                deri = self.feat(s, a) - sum(pi(s, b) * self.feat(s, b) for b in A)
              elif config.POLICY_TYPE == 'linear':
                # linear derivative
                deri = numpy.array([0,] * len(A))
                deri[A.index(a)] = 1
              else: raise Exception('unknown policy type')
              
              g = self.stepSize.getAlpha() * psi[rIdx] * futureRet * deri
              accG += g
              theta = theta + g

        # should for debug level.. to expensive to run
        #if config.VERBOSE: print self.computeObjValue(theta, psi, R, horizon, maxV)

        # stopping criteria
        if numpy.linalg.norm(accG) < 0.001: stopCounter += 1
        else: stopCounter = 0

        print numpy.linalg.norm(accG)

        if stopCounter > 50: break

      objValue = self.computeObjValue(theta, psi, R, horizon, maxV)
      if objValue > bestValue:
        bestTheta = theta
        bestValue = objValue
    
    if config.VERBOSE: print 'bestTheta', bestTheta
    optPi = self.thetaToOccupancy(bestTheta)
    
    if config.VERBOSE: print 'Sample', self.sampleTrajectory(optPi, s0, horizon, 'saPairs')
    return optPi

  def computeObjValue(self, theta, psi, R, horizon, maxV):
    ret = 0
    pi = self.thetaToOccupancy(theta)
    for rIdx in xrange(len(R)):
      rRet = self.computePiValue(pi, R[rIdx], horizon)
      if rRet > maxV[rIdx]:
        ret += psi[rIdx] * rRet
      else:
        ret += psi[rIdx] * maxV[rIdx]

    return ret
  
  def computePiValue(self, pi, r, horizon):
    """
    pi is generally stochastic. going to generate multiple trajectories to evaluate pi
    """
    ret = 0
    times = 10
    for _ in xrange(times):
      u = self.sampleTrajectory(pi, self.cmp.state, horizon, 'saPairs')
      ret += 1.0 * sum(r(s, a) for s, a in u) / times
    return ret


class SamplingAgent(PolicyGradientQueryAgent):
  def learn(self):
    args = easyDomains.convert(self.cmp, self.rewardSet, self.phi)
    rewardCandNum = len(args['R'])
    horizon = self.cmp.horizon
    k = config.NUMBER_OF_RESPONSES

    maxV = -numpy.inf
    maxQ = None

    for iterIdx in range(config.SAMPLE_TIMES):
      q = []
      for i in xrange(k):
        theta = [-0.5 + random.random() for _ in xrange(self.featLength)]
        q.append(self.thetaToOccupancy(theta))
      
      maxVs = []
      for rewardId in xrange(rewardCandNum):
        maxVs.append(max([self.computeV(pi, args['S'], args['A'], args['R'][rewardId], horizon) for pi in q]))
      objValue = sum(maxVs[idx] * self.phi[idx] for idx in range(rewardCandNum))
      if objValue > maxV:
        maxV = objValue
        maxQ = q
    
    return maxQ, None


class AprilAgent(PolicyGradientQueryAgent):
  """
  Randomly partition psi and find their optimal policies
  Repeat this for finite number of times, specified by ?
  
  only works for k == 2
  """
  def learn(self):
    args = easyDomains.convert(self.cmp, self.rewardSet, self.phi)
    rewardCandNum = len(args['R'])
    k = config.NUMBER_OF_RESPONSES
    assert k == 2 # not going to work for k > 2

    maxV = -numpy.inf
    maxQ = None
    for iterIdx in range(config.SAMPLE_TIMES):
      selector = [random.random() > .5 for _ in xrange(rewardCandNum)]
      # got two psis
      psi0 = [self.phi[_] if selector[_] else 0 for _ in xrange(rewardCandNum)]
      psi1 = [self.phi[_] if not selector[_] else 0 for _ in xrange(rewardCandNum)]
      agent0 = self.getFiniteVIAgent(psi0, self.cmp.horizon - self.cmp.getResponseTime(), self.cmp.terminalReward, posterior=True)
      agent1 = self.getFiniteVIAgent(psi1, self.cmp.horizon - self.cmp.getResponseTime(), self.cmp.terminalReward, posterior=True)
      
      v = agent0.getValue(self.cmp.state) + agent1.getValue(self.cmp.state) 
      if v > maxV:
        maxV = v
        maxQ = [agent0.x, agent1.x]

    return maxQ, None


class PolicyGradientRandQueryAgent(PolicyGradientQueryAgent):
  def findNextPolicy(self, S, A, R, T, s0, psi, maxV):
    # randomly generate a policy to add to the query
    theta = [-0.5 + random.random() for _ in xrange(self.featLength)] # baseline
    optPi = self.thetaToOccupancy(theta)

    return optPi


class PolicyGradientAgent(ValueIterationAgent):
  """
  Policy gradient to solve a policy for a given reward function.
  Implemented in a way that calls policy gradient query agent, which has one reward candidate
  """
  def __init__(self, mdp, feat, featLength, discount = 1.0, horizon = numpy.inf):
    ValueIterationAgent.__init__(self, mdp, discount)
    self.feat = feat
    self.featLength = featLength
    self.horizon = horizon

  def learn(self):
    rewardSet = [self.mdp.getReward]
    psi = [1]

    # the agent is certain on the reward functions
    args = easyDomains.convert(self.mdp, rewardSet, psi)
    args['maxV'] = [-numpy.inf]

    self.agent = PolicyGradientQueryAgent(self.mdp, rewardSet, psi, QueryType.POLICY, self.feat, self.featLength, self.discount)
    self.optPi = self.agent.findNextPolicy(**args)
    self.x = lambda s, a: self.optPi(s, a)

    return self.optPi

  def getValue(self, state, t=0):
    return self.agent.computePiValue(self.optPi, self.mdp.getReward, self.horizon)