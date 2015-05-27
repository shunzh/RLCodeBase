from learningAgents import ReinforcementAgent
import numpy as np
import util

class PolicyIterationAgent(ReinforcementAgent):
  """
  Policy iteration
  """
  def __init__(self, mdp, **args):
    """
    Args:
      mdp: mdp used for training
      threshold: if changes in V smaller than threshold, then terminate policy estimation.
    """
    ReinforcementAgent.__init__(self, **args)

    # model based 
    self.mdp = mdp

    try:
      self.threshold = args['threshold']
    except:
      self.threshold = .5 # default threshold

    self.values = util.Counter()
    self.policies = util.Counter()
  
  def getQValue(self, state, action):
    possibleNextStates = self.mdp.getTransitionStatesAndProbs(state, action)
    return sum([(self.mdp.getReward(state, action, nextState) + self.gamma * self.getValue(nextState)) * prob\
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
        actions = self.mdp.getPossibleActions(state)
        self.policies[state] = max(actions, key = lambda a: self.getQValue(state, a))
        
        if b != self.policies[state]: policyStable = False
      
      return policyStable
    
    policyStable = False
    while (not policyStable):
      policyEstimate()
      policyStable = policyImprovement()

  def getValue(self, state):
    return self.values[state]

  def getAction(self, state):
    return self.getPolicy(state)

  def getPolicy(self, state):
    if state in self.policies.keys():
      return self.policies[state]
    else:
      # return the first action if not learned
      return self.mdp.getPossibleActions(state)[0]