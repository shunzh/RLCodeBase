import numpy as np
from inverseRL import InverseRL
from policyIterationAgents import PolicyIterationAgent
import cma

class InverseBayesianRL(InverseRL):
  """
  P(R|samples) ~ P(samples|R) P(R)

  Ramachandran, Deepak, and Eyal Amir.
  "Bayesian inverse reinforcement learning."
  Urbana 51 (2007): 61801.
  """
  def __init__(self, mdp, rewardPrior, eta = 1, stepSize = 1):
    """
    Args:
      rewardPrior: P(R)
      stepSize: \sigma in the paper, the granularity of reward space
    """
    InverseRL.__init__(self, eta)

    self.rewardPrior = rewardPrior
    self.stepSize = 1
    self.n = len(mdp.getStates())
    
    if not "setReward" in dir(mdp):
      raise Exception("setReward not found in MDP " + mdp.__class__.__name__ + ". \
This is necessary in bayesian irl")
    
    opts = {'gamma': 0.8, 
            'actionFn': lambda state: mdp.getPossibleActions(state)}
    self.agent = PolicyIterationAgent(mdp, **opts)
  
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
    qFunc = lambda s, a: self.agent.getQValue(s, a)
    likelihood = self.softMaxSum(qFunc)
    
    # want to minimize
    return - (priorProb + likelihood)
  
  def solve(self):
    start_pos = [0] * self.n
    
    result = cma.fmin(self.obj, start_pos, 1)

    print result[0]