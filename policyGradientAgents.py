from QTPAgent import GreedyConstructionPiAgent
import numpy
import random
import config
from valueIterationAgents import ValueIterationAgent
import easyDomains
from cmp import QueryType

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
    return self.initAlpha / self.k
  
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
    GreedyConstructionPiAgent.__init__(self, cmp, rewardSet, initialPhi, queryType, gamma)
    self.feat = feat
    self.featLength = featLength
    self.stepSize = ConstantStepSize(0.01)

  def thetaToOccupancy(self, theta):
    def getActProb(s, a):
      maxV = max(numpy.dot(theta, self.feat(s, b)) for b in self.args['A'])
      actProbs = {b: numpy.exp(numpy.dot(theta, self.feat(s, b)) - maxV) for b in self.args['A']}
      return actProbs[a] / sum(actProbs.values())
    
    return getActProb

  def findNextPolicy(self, S, A, R, T, s0, psi, maxV):
    """
    Same arguments as lp.milp
    Return: next policy to add. It's a parameter, not occupancy
    
    FIXME not re-using the code in PolicyGradientAgent. they are very similar classes. shall we?
    """
    # start with a `trivial' controller
    horizon = self.cmp.horizon
    
    # compute the derivative of EUS
    for rspTime in xrange(1):
      theta = [0] * self.featLength

      self.stepSize.reset()

      for iterStep in xrange(1000):
        pi = self.thetaToOccupancy(theta)
        # u is a list of state action pairs
        # this is still policy query.. we sample to the task horizon
        u = self.sampleTrajectory(pi, s0, horizon, 'saPairs')
        self.stepSize.iterate()
        if config.DEBUG: print iterStep, u

        for rIdx in range(len(R)):
          ret = sum(R[rIdx](s, a) for s, a in u)
          if config.DEBUG: print 'ret in reward id', rIdx, 'is', ret

          if ret > maxV[rIdx]:
            # here is where the non-smoothness comes from
            # only add the derivative when the accumulated return is larger than the return obtained by the
            # best policy in the query set
            futureRet = 0
            for s, a in reversed(u):
              futureRet += R[rIdx](s, a)
              deri = self.feat(s, a) - sum(pi(s, b) * self.feat(s, b) for b in self.args['A'])
              theta = theta + self.stepSize.getAlpha() * psi[rIdx] * futureRet * deri
    
    #theta = [-0.5 + random.random() for _ in xrange(self.featLength)] # baseline
    optPi = self.thetaToOccupancy(theta)
    
    #print bestTheta
    if config.VERBOSE:
      for _ in xrange(3):
        print 'Sample #', _, self.sampleTrajectory(optPi, s0, horizon, 'saPairs')
    return optPi

  def computeObjValue(self, theta, psi, R, horizon, maxV):
    ret = 0
    pi = self.thetaToOccupancy(theta)
    for rIdx in xrange(len(R)):
      rRet = self.computePiValue(pi, R[rIdx], horizon)
      if rRet > maxV[rIdx]:
        ret += psi[rIdx] * rRet

    return ret
  
  def computePiValue(self, pi, r, horizon):
    """
    pi is generally stochastic. going to generate multiple trajectories to evaluate pi
    """
    ret = 0
    times = 5
    for _ in xrange(times):
      u = self.sampleTrajectory(pi, self.cmp.state, horizon, 'saPairs')
      ret += 1.0 * sum(r(s, a) for s, a in u) / times
    return ret


class PolicyGradientAgent(ValueIterationAgent):
  """
  Policy gradient to solve a policy for a given reward function.
  Implemented in a way that calls policy gradient query agent, which has one reward candidate
  """
  def __init__(self, mdp, feat, featLength, discount = 1.0):
    ValueIterationAgent.__init__(self, mdp, discount)
    self.feat = feat
    self.featLength = featLength

  def learn(self):
    rewardSet = [self.mdp.getReward]
    psi = [1]

    # the agent is certain on the reward functions
    args = easyDomains.convert(self.mdp, rewardSet, psi)
    args['maxV'] = [-numpy.inf]

    a = PolicyGradientQueryAgent(self.mdp, rewardSet, psi, QueryType.POLICY, self.feat, self.featLength, self.discount)
    return a.findNextPolicy(**args)
