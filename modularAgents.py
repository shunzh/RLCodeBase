from qlearningAgents import ApproximateQAgent

import numpy as np

class ModularAgent(ApproximateQAgent):
  def __init__(self, **args):
    ApproximateQAgent.__init__(self, **args)
 
  def getQValue(self, state, action):
    """
    Get Q value by consulting each module.
    """
    # sum over q values from each sub mdp
    return sum([self.getSubQValue(state, action, i) for i in xrange(self.nModules)])
  
  def getSubQValue(self, s, a, moduleId):
    state, action = self.mapper(s, a)
    return self.qFuncs[moduleId](state, action, self.para)

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
  
  def setParameters(self, x):
    """
    Parameters used to config the Q
    """
    self.para = x

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
