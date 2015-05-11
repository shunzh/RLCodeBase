from learningAgents import ValueEstimationAgent
import numpy as np
import util

class PolicyIterationAgent(ValueEstimationAgent):
  """
  Policy iteration
  """
  def __init__(self, mdp, threshold = .5):
    """
    Args:
      mdp: mdp used for training
      threshold: if changes in V smaller than threshold, then terminate policy estimation.
    """
    self.mdp = mdp
    self.threshold = threshold

    self.values = util.Counter()
    self.policies = util.Counter()
  
  def getQValue(self, state, action):
    possibleNextStates = self.mdp.getTransitionStatesAndProbs(state, action)
    return sum([(self.mdp.getReward(nextState) + self.getValue(nextState)) * prob\
                for nextState, prob in possibleNextStates])

  def learn(self):
    """
    Run policy estimation and policy improvement until convergence.
    """
    def policyEstimate():
      delta = np.inf
      while (delta > self.threshold):
        delta = 0

        for state in self.mdp.getStates():
          v = self.values[state]

          action = self.getPolicy(state)
          self.values[state] = self.getQValue(state, action)
          delta = max(delta, abs(self.values[state] - v))

    def policyImprovement():
      policyStable = True
      for state in self.mdp.getStates():
        b = self.getPolicy(state)
        actions = self.mdp.getActions(state)
        self.policies[state] = max(actions, key = lambda a: self.getQValue(state, a))
        
        if b != self.policies[state]: policyStable = False
      
      return policyStable
    
    policyStable = False
    while (not policyStable):
      policyEstimate()
      policyStable = policyImprovement()

  def getValue(self, state):
    return self.values[state]

  def getPolicy(self, state):
    return self.policies[state]