from mdp import MarkovDecisionProcess
from valueIterationAgents import ValueIterationAgent
import numpy

# possible type of queries
# TODO some are not implemented
class QueryType:
  POLICY, REWARD, REWARD_SIGN, NONE = range(4)
  
class ControlledMarkovProcess(MarkovDecisionProcess):
  def __init__(self, queries, trueReward, gamma, responseTime, horizon=numpy.inf, terminalReward=None):
    MarkovDecisionProcess.__init__(self)
    
    self.outsandingQuery = None
    # possible queries
    self.queries = queries
    self.responseTime = responseTime
    
    # let agent know use finite or infinite horizon vi
    self.horizon = horizon
    self.terminalReward = terminalReward
    
    # this field is needed for reward queries
    self.possibleRewardValues = None
    
    # the real reward function
    # learn a VI agent on this reward setting, and policy will be decided.
    self.getReward = trueReward
    self.viAgent = ValueIterationAgent(self, gamma)
    self.viAgent.learn()

  def setPossibleRewardValues(self, rewards):
    self.possibleRewardValues = rewards

  def query(self, q):
    """
    q = (QueryType, state)
    type = ['policy','reward','rewardSign']
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
    """
    if self.outsandingQuery != None and self.timer >= self.outsandingQuery[1]:
      # issues a response
      type, s = self.outsandingQuery[0]
      if type == QueryType.POLICY:
        res = self.viAgent.getPolicy(s, self.timer)
      elif type == QueryType.REWARD:
        res = self.getReward(s)
      elif type == QueryType.NONE:
        res = 0 # return a dummy value
      else:
        raise Exception('Unkown type of query ' + str(type))
      
      self.outsandingQuery = None
      return res
    else: return None