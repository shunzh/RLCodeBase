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

    # assume the weights are not dynamically learned, intialize them here.
    self.weights = [0, 0, 1]
    self.learningWeights = False
 
  def getQValue(self, state, action):
    """
    Get Q value by consulting each module
    """
    # sum over q values from each sub mdp
    return sum([self.qFuncs[i](state, action) * self.weights[i] for i in xrange(len(self.qFuncs))])

  def getSubQValues(self, state, action):
    return [self.qFuncs[i](state, action) for i in xrange(len(self.qFuncs))]

  def getSoftmaxQValue(self, state, action):
    actions = self.getLegalActions(state)
    j = actions.index(action)

    # q value matrix
    # vMat(i, j) is for i-th module and j-th action
    vMat = []

    # iterate through all the modules
    for qFunc in self.qFuncs:
      # list of exp^q
      exps = [math.exp(qFunc(state, action)) for action in actions]

      # Normalize
      sumExps = sum(exps)
      vMat.append([exp / sumExps for exp in exps])

    # sum over j-th column (j-th action)
    return sum([vMat[i][j] for i in range(len(self.qFuncs))])

  def setQFuncs(self, qFuncs):
    """
    Set QFuncs from the environment. getQValue will use this.
    """
    self.qFuncs = qFuncs

  def setWeights(self, weights):
    """
    If this is set, then the agent simply follows the weights.
    """
    self.learningWeights = False
    self.weights = weights

  def getWeights(self):
    """
    Get weights.
    """
    return self.weights
  
  def update(self, state, action, nextState, reward):
    if not self.learningWeights:
      pass
    else:
      # TODO
      raise Exception("calling unimplemented method: update for learning weight.")

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

  def getAction(self, state):
    return ModularAgent.getAction(self, self.getState(state))

  def update(self, state, action, nextState, reward):
    return ModularAgent.update(self, self.getState(state), action, self.getState(nextState), reward)

  def final(self, state):
    return ModularAgent.final(self, self.getState(state))

