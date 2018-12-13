from learningAgents import ReinforcementAgent
import numpy as np
import util

class PolicyIterationAgent:
  """
  Policy iteration
  """
  def __init__(self, mdp, threshold=.1):
    """
    Args:
      mdp: mdp used for training
      threshold: if changes in V smaller than threshold, then terminate policy estimation.
    """
    # model based 
    self.mdp = mdp

    self.threshold = threshold

    self.values = util.Counter()
    self.policies = util.Counter()
  
  def getQValue(self, state, action):
    return self.mdp['r'][state, action] +\
           self.mdp['gamma'] * sum(self.mdp['T'][state, action, nextState] * self.getValue(nextState) for nextState in self.mdp['S'])

  def policyEstimate(self):
    delta = np.inf
    while (delta > self.threshold):
      delta = 0

      for state in self.mdp['S']:
        v = self.values[state]

        action = self.getPolicy(state)
        self.values[state] = self.getQValue(state, action)
        delta = max(delta, abs(self.values[state] - v))

  def policyImprovement(self):
    policyStable = True
    for state in self.mdp['S']:
      b = self.getPolicy(state)
      actions = self.mdp['A']
      self.policies[state] = max(actions, key = lambda a: self.getQValue(state, a))
      
      if b != self.policies[state]: policyStable = False

    return policyStable
    
  def learn(self):
    """
    Run policy estimation and policy improvement until convergence.
    """
    policyStable = False
    while (not policyStable):
      self.policyEstimate()
      policyStable = self.policyImprovement()

  def getValue(self, state):
    return self.values[state]

  def getAction(self, state):
    return self.getPolicy(state)

  def getPolicy(self, state):
    if state in self.policies.keys():
      return self.policies[state]
    else:
      # return the first action if not learned
      return self.mdp['A'][0]