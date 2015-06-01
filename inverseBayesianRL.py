import numpy as np
from inverseRL import InverseRL
from policyIterationAgents import PolicyIterationAgent
import random
import config

class InverseBayesianRL(InverseRL):
  """
  P(R|samples) ~ P(samples|R) P(R)

  Ramachandran, Deepak, and Eyal Amir.
  "Bayesian inverse reinforcement learning."
  Urbana 51 (2007): 61801.
  """
  def __init__(self, mdp, rewardPrior, eta = 1, stepSize = 1, lastWindow = 100):
    """
    Args:
      rewardPrior: P(R)
      eta: parameter for softmax
      stepSize: \sigma in the paper, the granularity of reward space
      maxIterations: the number of iterations that policy walk will run
      lastWindow: the rewards in the last iterations will be considered
    """
    InverseRL.__init__(self, eta)

    self.rewardPrior = rewardPrior
    self.stepSize = 1
    self.n = len(mdp.getStates())
    
    self.lastWindow = lastWindow
    
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
    
    return priorProb + likelihood
  
  def solve(self):
    r = [0] * self.n
    stableSteps = 0
    
    while True:
      p = self.obj(r)

      # randomly choose a neighbor
      idx = random.randint(0, self.n - 1)
      diff = random.choice([+self.stepSize, -self.stepSize])
      r[idx] += diff
      
      newP = self.obj(r)
      
      walkProb = min(1, np.exp(newP - p))
      
      # walk to new r with prob of walkProb, otherwise revert
      if random.random() >= walkProb:
        r[idx] -= diff
      else:
        stableSteps += 1
      
      if config.DEBUG:
        print "Iteration ", _, ": reward ", r

    return r
