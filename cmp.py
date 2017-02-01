from mdp import MarkovDecisionProcess
from valueIterationAgents import ValueIterationAgent
import numpy as np
from lp import computeValue

# possible type of queries
# TODO some are not implemented
class QueryType:
  ACTION, REWARD, REWARD_SIGN, POLICY, PARTIAL_POLICY, DEMONSTRATION, COMMITMENT,\
  SIMILAR, NONE = range(9)
  
class ControlledMarkovProcess(MarkovDecisionProcess):
  def __init__(self, responseTimes, horizon=np.inf, terminalReward=None):
    MarkovDecisionProcess.__init__(self)
    
    self.outsandingQuery = None
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

  def decorate(self, gamma, queries):
    self.gamma = gamma
    
    # possible queries
    self.queries = queries

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
  
  def getStateDistance(self, s1, s2):
    """
    Return the distance between two states
    """
    raise Exception('undefined')
  
  def getTrajectoryDistance(self, u1, u2):
    assert len(u1) == len(u2)
    return sum(self.getStateDistance(s1, s2) for (s1, s2) in zip(u1, u2))

  def responseCallback(self, agent):
    """
    The agent should check with this function to see whether there is a feedback
    This is just for the experiment.
    """
    # FIXME not paying attention here when refactoring
    if self.outsandingQuery != None and self.timer >= self.outsandingQuery[1]:
      # issues a response
      q = self.outsandingQuery[0]
      resSet, consistCond = agent.getConsistentCond(q)
      
      # we assumed that the first consistent response is returned
      for res in resSet:
        if consistCond(res, self.rewardIdx):
          self.outsandingQuery = None
          return res
    else: return None