import numpy as np
from inverseRL import InverseRL
from policyIterationAgents import PolicyIterationAgent

class InverseBayesianRL(InverseRL):
  """
  P(R|samples) ~ P(samples|R) P(R)

  Ramachandran, Deepak, and Eyal Amir.
  "Bayesian inverse reinforcement learning."
  Urbana 51 (2007): 61801.
  """
  def __init__(self, mdp, rewardPrior, stepSize = 1):
    """
    Args:
      rewardPrior: P(R)
      stepSize: \sigma in the paper, the granularity of reward space
    """
    self.mdp = mdp
    self.rewardPrior = rewardPrior
    self.stepSize = 1
    
    self.agent = PolicyIterationAgent(mdp)
  
  def obj(self, X):
    """
    Args:
      X: vector of reward
    Return:
      log(P(R|samples))
    """
    priorProb = sum(np.log(self.rewardPrior(r)) for r in X)

    for state, reward in zip(self.agent.mdp.getStates(), X):
      self.agent.mdp.setReward(state, reward)
    self.agent.learn()
    qFunc = lambda s, a: self.getQValue(s, a)
    likelihood = self.softMaxSum(qFunc)
    
    return priorProb + likelihood
  
  def solve(self):
    # TODO

    # initialize the reward to be all 0s
    r = [0] * len(self.mdp.getStates())