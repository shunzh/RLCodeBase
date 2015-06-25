from modularAgents import ModularAgent

class RandomAgent(ModularAgent):
  def getValue(self, state):
    return 0.0

  def getQValue(self, state, action):
    return 0.0
  

class ReflexAgent(ModularAgent):
  """
  Same setting as modular agent.
  Instead of weighted sum of Q functions, just select the one with the largest Q value.
  That is, it always esponds to the most promising module.
  """
  def getQValue(self, state, action):
    """
    Override so that q value with the largest magnitude is returned.
    Policy hereby reflects the module with the largest q value.
    """
    maxMagnitude = 0
    maxIdx = -1
    for i in xrange(self.nModules):
      magnitude = abs(self.getSubQValue(state, action, i))
      if magnitude > maxMagnitude:
        maxMagnitude = magnitude
        maxIdx = i
        
    return self.getSubQValue(state, action, maxIdx)