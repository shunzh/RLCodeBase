from mdp import MarkovDecisionProcess
from valueIterationAgents import ValueIterationAgent

class ControlledMarkovProcess(MarkovDecisionProcess):
  def __init__(self, queries, trueReward):
    MarkovDecisionProcess.__init__(self)
    
    self.outsandingQueries = None
    # possible queries
    self.queries = queries
    
    # the real reward function
    # learn a VI agent on this reward setting, and policy will be decided.
    self.getReward = trueReward
    self.viAgent = ValueIterationAgent(self)

  def query(self, q):
    """
    Ask the policy of this state
    """
    self.outsandingQueries = (q, self.timer + self.getResponseTime)
  
  def cost(self, q):
    """
    Return cost of querying this given q
    """
    abstract
  
  def responseCallback(self):
    """
    The agent should check with this function to see whether there is a feedback
    For now, it is not called, as we only do in mind simulation. 
    """
    if self.timer >= self.outsandingQueries[1]:
      # issues a response
      res = self.viAgent.getPolicy(self.outsandingQueries[0])
    
    self.outsandingQueries = None
    return res