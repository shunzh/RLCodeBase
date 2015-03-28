import random
from modularAgents import ModularAgent

class RandomAgent:
  def __init__(self, mdp):
    self.mdp = mdp

  def getAction(self, state):
    return self.getPolicy(state)

  def getValue(self, state):
    return 0.0

  def getQValue(self, state, action):
    return 0.0

  def getPolicy(self, state):
    return random.choice(self.mdp.getPossibleActions(state))

  def update(self, state, action, nextState, reward):
    pass      


class ReflexAgent(ModularAgent):
  """
  Same setting as modular agent.
  Instead of weighted sum of Q functions, just select the one with the largest Q value.
  That is, it always esponds to the most promising module.
  """
  def getQValue(self, state, action):
    """
    Override so that max q value is returned.
    Policy hereby reflects the module with the largest q value.
    """
    return max([self.qFuncs[i](state, action) for i in xrange(self.nModules)])


class RoundRobinAgent(ModularAgent):
  """
  Same setting as modular agent.
  Selecting module in a Round-Robin way. 
  """
  def __init__(self, **args):
    ModularAgent.__init__(self, **args)
    
    # policy is time-dependent, need to record the time
    self.timer = 0
    # how long we keep one module on
    self.timePerModule = 5
  
  def update(self, state, action, nextState, reward):
    ModularAgent.update(self, state, action, nextState, reward)

    self.timer += 1

  def getPolicy(self, state):
    moduleId = (self.timer / self.timePerModule) % self.nModules
    actions = self.getLegalActions(state)

    q_value_func = lambda action: self.getSubQValues(state, action)['subs'][moduleId]
    maxQValue = max([q_value_func(action) for action in actions])
    optActions = [action for action in actions if q_value_func(action) == maxQValue]
    return random.choice(optActions)