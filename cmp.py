from mdp import MarkovDecisionProcess
from valueIterationAgents import ValueIterationAgent

class ControlledMarkovProcess(MarkovDecisionProcess):
  def __init__(self, queries, trueReward, gamma):
    MarkovDecisionProcess.__init__(self)
    
    self.outsandingQuery = None
    # possible queries
    self.queries = queries
    
    # the real reward function
    # learn a VI agent on this reward setting, and policy will be decided.
    self.getReward = trueReward
    self.viAgent = ValueIterationAgent(self, gamma)
    self.viAgent.learn()

  def query(self, q):
    """
    Ask the policy of this state
    """
    self.outsandingQuery = (q, self.timer + self.responseTime)
  
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
    if self.outsandingQuery != None and self.timer >= self.outsandingQuery[1]:
      # issues a response
      res = self.viAgent.getPolicy(self.outsandingQuery[0])
      self.outsandingQuery = None
      return res
    else: return None