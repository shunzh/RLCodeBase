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
    return sum([self.qFuncs[i](state, action, self.discounters) * self.weights[i] for i in xrange(self.nModules)])

  def getSubQValues(self, state, action):
    """
    Return decomposed Q value for debugging.
    """
    return {'sum': self.getQValue(state, action),\
            'subs': [self.qFuncs[i](state, action) for i in xrange(self.nModules)]}

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
    there is no learning here
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


class ReducedModularAgent(ModularAgent):
  """
  Modular agent, which first maps state to an inner representation (so the state space reduced).
  """
  def __init__(self, **args):
    ModularAgent.__init__(self, **args)

    # Set get state function here.
    # By default, it is an identity function
    self.getState = lambda x : x

  def setStateFilter(self, extractor):
    """
    Set the state filter here, which returns the state representation for learning.
    The default one is an identity function.
    """
    self.getState = extractor

  def getSubQValues(self, state, action):
    return ModularAgent.getSubQValues(self, self.getState(state), action)

  def getAction(self, state):
    return ModularAgent.getAction(self, self.getState(state))

  def update(self, state, action, nextState, reward):
    return ModularAgent.update(self, self.getState(state), action, self.getState(nextState), reward)

  def final(self, state):
    return ModularAgent.final(self, self.getState(state))

