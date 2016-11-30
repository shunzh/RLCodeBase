from QTPAgent import GreedyConstructionPiAgent
import numpy
import random

class DiminishingStepSize:
  def __init__(self, alpha):
    self.initAlpha = alpha
    self.k = 0.0
  
  def getAlpha(self):
    self.k += 1
    return self.initAlpha / self.k
  
  def reset(self):
    self.k = 0.0


class PolicyGradientQueryAgent(GreedyConstructionPiAgent):
  """
  This finds the next policy by gradient descent using EUS as the objective function
  """
  def __init__(self, cmp, rewardSet, initialPhi, queryType, feat, featLength, gamma):
    GreedyConstructionPiAgent.__init__(self, cmp, rewardSet, initialPhi, queryType, gamma)
    self.feat = feat
    self.featLength = featLength
    self.stepSize = DiminishingStepSize(1)

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
    bestTheta = [0] * self.featLength
    bestObjValue = -numpy.inf
    
    # compute the derivative of EUS
    for rspTime in xrange(2):
      theta = [-0.5 + random.random() for _ in xrange(self.featLength)]
      self.stepSize.reset()
      #theta = [0] * self.featLength
      for iterStep in xrange(50):
        pi = self.thetaToOccupancy(theta)
        # u is a list of state action pairs
        # this is still policy query.. we sample to the task horizon
        u = self.sampleTrajectory(pi, s0, horizon, 'saPairs')

        #debug
        #print 'pi', pi((0, 0), 1), pi((0, 0), 0), pi((0, 0), -1)
        #print theta
        #print u

        for rIdx in range(len(R)):
          ret = sum(R[rIdx](s, a) for s, a in u)

          if ret > maxV[rIdx]:
            # here is where the non-smoothness comes from
            # only add the derivative when the accumulated return is larger than the return obtained by the
            # best policy in the query set
            futureRet = 0
            for s, a in reversed(u):
              futureRet += R[rIdx](s, a)
              deri = self.feat(s, a) - sum(pi(s, b) * self.feat(s, b) for b in self.args['A'])
              theta = theta + self.stepSize.getAlpha() * psi[rIdx] * futureRet * deri

              # compute the obj function using theta
              objValue = self.computeObjValue(theta, psi, R, horizon, maxV)

              if objValue > bestObjValue:
                bestObjValue = objValue
                bestTheta = theta
    
    optPi = self.thetaToOccupancy(bestTheta)
    
    #print bestTheta
    print self.sampleTrajectory(optPi, s0, horizon, 'saPairs')
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
    ret = 0
    for _ in xrange(10):
      u = self.sampleTrajectory(pi, self.cmp.state, horizon, 'saPairs')
      ret += sum(r(s, a) for s, a in u) / 10
    return ret


