from scipy.optimize import minimize
import numpy as np

class InverseRL:
  """
  Inverse Reinforcement Learning algorithm.

  This is implemented as
  Ng, Andrew Y., and Stuart J. Russell.
  "Algorithms for inverse reinforcement learning."
  ICML. 2000.
  """
  def setSamplesFromMdp(self, mdp, agent):
    """
    One way to set the samples.
    Read form mdp and get the policies of states.

    Set self.getSamples and self.getActions here.
    """
    states = mdp.getStates()
    self.getSamples = lambda : [(state, agent.getPolicy(state)) for state in states]
    self.getActions = lambda s : mdp.getPossibleActions(s)

  def setSamples(self, samples, actions):
    """
    Read from subj*.parsed.mat file.

    Set self.getSamples and self.getActions here.
    """
    self.getSamples = lambda : samples
    self.getActions = lambda s : actions

  def printSamples(self):
    samples = self.getSamples()

    for state, action in samples:
      print state
      print action
 
  def softMaxSum(self, qFunc):
    """
    Measures the probability of observing such policies given the q function.
    """
    ret= 0
    
    # replay the process
    for state, optAction in self.getSamples():
      def qToPower(v):
        # v -> exp(eta * v)
        return np.exp(self.eta * v)

      actionSet = self.getActions(state)
      qValues = {action: qFunc(state, action) for action in actionSet}

      # both numerator and denominator raise to the power of e
      # numerator: optimal action
      ret += qValues[optAction]
      # denominator: all the actions
      ret -= np.log(sum([qToPower(qValues[action]) for action in actionSet]))

    # This is to be minimized, take the negative.
    return - ret

  def pSum(self, qFunc):
    """
    Using p function defined in Ng's paper.
    """
    ret= 0
    
    for state, optAction in self.getSamples():
      actionSetExceptOpt = self.getActions(state).remove(optAction)
      qValuesExceptOpt = [qFunc(state, action) for action in actionSetExceptOpt]
      qValuesExceptOpt.sort()

      p = lambda x: x if x >= 0 else 2 * x
      
      ret += p(qFunc(state, optAction) - qValuesExceptOpt[0])
      
    # This is to be minimized, take the negative.
    return - ret

  def obj(self, X):
    """
    Args:
      X: parameter vector. weights on basis of rewards
    Return:
      The objective function value
    """
    alpha = X

    def getReward(state, action, nextState):
      # dot(alpha, nextState)
      return sum([coef * feat for coef, feat in zip(alpha, nextState)])

    #TODO what is q function?
    def computeQValue(state, action):
      return None

    return self.pSum(computeQValue)