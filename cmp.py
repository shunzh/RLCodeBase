from mdp import MarkovDecisionProcess

class ControlledMarkovProcess(MarkovDecisionProcess):
  def __init__(self):
    MarkovDecisionProcess.__init__(self)
    
    self.outsandingQueries = None

  def getGroundtruthPolicy(self, state):
    """
    Return the ground truth policy
    """
    abstract

  def getResponseTime(self, state):
    """
    Return expected query time
    """
    abstract
  
  def setRewardAssumptions(self, rewards):
    abstract
  
  def queryPolicy(self, state):
    """
    Ask the policy of this state
    """
    self.outsandingQueries = (state, self.timer + self.getResponseTime(state))
  
  def responseCallback(self):
    if self.timer >= self.outsandingQueries[1]:
      # issues a response
      policy = self.getGroundtruthPolicy(self.outsandingQueries[0])
    
    self.outsandingQueries = None
    return policy