import numpy as np
from inverseRL import InverseRL
from policyIterationAgents import PolicyIterationAgent
import random

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
    self.rewardPrior = rewardPrior
    self.stepSize = 1
    self.n = len(mdp.getStates())
    
    if not "setReward" in dir(mdp):
      raise Exception("setReward not found in MDP " + mdp.__class__.__name__ + ". \
This is necessary in bayesian irl")
    
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
    """
    Implement the PolicyWalk algorithm in the paper.
    """
    # initialize the reward to be all 0s
    r = [0] * self.n
    
    while True:
      p = self.obj(r)

      # randomly choose a neighbor
      idx = random.randint(0, self.n)
      diff = random.choice(+self.stepSize, -self.stepSize)
      r[idx] += diff
      
      newP = self.obj(r)
      
      walkProb = min(1, np.exp(newP - p))
      
      # walk to new r with prob of walkProb, otherwise revert
      if random.random() >= walkProb:
        r[idx] -= diff