from qlearningAgents import ApproximateQAgent

import math
import numpy as np

class ModularAgent(ApproximateQAgent):
  """
  State: location of the agent.
  Action: weights on the sub-MDPs.
  Transition: transition of the agent.
  Reward: reward from the environment.

  Assume:
  Weights are independent from the location of the agent.
  """
  def __init__(self, **args):
    ApproximateQAgent.__init__(self, **args)

    # weights and discounters should be set later after learning
    self.weights = [0, 0, 1]
    self.discounters = [0.6] * 3
 
  def getQValue(self, state, action):
    """
    Get Q value by consulting each module.
    """
    # sum over q values from each sub mdp
    return sum([self.getSubQValue(state, action, i) * self.weights[i] for i in xrange(self.nModules)])
  
  def getSubQValue(self, s, a, moduleId):
    state, action = self.mapper(s, a)
    return self.qFuncs[moduleId](state, action, self.discounters)

  def getSubQValues(self, state, action):
    """
    Return decomposed Q value for debugging.
    """
    return {'sum': self.getQValue(state, action),\
            'subs': [self.getsubqvalue(state, action, i) for i in xrange(self.nModules)]}

  def setQFuncs(self, qFuncs):
    """
    Set QFuncs from the environment. getQValue will use this.
    """
    self.qFuncs = qFuncs
    self.nModules = len(self.qFuncs)

  def setWeights(self, weights):
    self.weights = weights
  
  def setDiscounters(self, discounters):
    self.discounters = discounters

  def getWeights(self):
    return self.weights
  
  def update(self, state, action, nextState, reward):
    """
    There is no learning here.
    Don't even run the update in base class.
    """
    pass

  def getSignificance(self, state):
    """
    How significance an agent's correct decision at this state should affect the overall performance.

    Using the std of the Q values of this state.

    DUMMY
    """
    actions = self.getLegalActions(state)

    values = [self.getQValue(state, action) for action in actions]
    return np.std(values)

