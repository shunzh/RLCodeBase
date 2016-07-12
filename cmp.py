from mdp import MarkovDecisionProcess
from valueIterationAgents import ValueIterationAgent
import numpy as np

# possible type of queries
# TODO some are not implemented
class QueryType:
  ACTION, REWARD, REWARD_SIGN, POLICY, PREFERENCE, NONE = range(6)
  
class ControlledMarkovProcess(MarkovDecisionProcess):
  def __init__(self, queries, trueReward, gamma, responseTimes, horizon=np.inf, terminalReward=None):
    MarkovDecisionProcess.__init__(self)
    
    self.outsandingQuery = None
    # possible queries
    self.queries = queries
    if type(responseTimes) == int: 
      # deterministic response time
      self.responseTimes = [(responseTimes, 1)]
    elif type(responseTimes) == list:
      self.responseTimes = responseTimes
    else:
      raise Exception('unknown type of responeTimes')
    
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

  def getResponseTime(self):
    # TODO stochastic case
    return self.responseTimes[0][0]

  def query(self, q):
    """
    q = (type, state)
    type \in {'policy','reward','rewardSign'}
    """
    self.outsandingQuery = (q, self.timer + self.getResponseTime())
  
  def cost(self, q):
    """
    Return cost of querying this given q
    """
    raise Exception('undefined')
  
  def responseCallback(self):
    """
    The agent should check with this function to see whether there is a feedback
    """
    if self.outsandingQuery != None and self.timer >= self.outsandingQuery[1]:
      # issues a response
      type, s = self.outsandingQuery[0]
      if type == QueryType.ACTION:
        # give policy at the response time
        res = self.viAgent.getPolicy(s, self.timer)
      elif type == QueryType.REWARD:
        res = self.getReward(s)
      elif type == QueryType.REWARD_SIGN:
        res = np.sign(self.getReward(s))
      elif type == QueryType.NONE:
        res = 0 # return a dummy value
      else:
        raise Exception('Unkown type of query ' + str(type))
      
      self.outsandingQuery = None
      return res
    else: return None