from cmp import ControlledMarkovProcess

class ChainDomain(ControlledMarkovProcess):
  def __init__(self, queries, rewardFunc, length=5):
    self.length = length
    self.responseTime = 2

    ControlledMarkovProcess.__init__(self, queries, rewardFunc)
  
  def getStates(self):
    return range(self.length)
  
  def getStartState(self):
    return 3
  
  def getPossibleActions(self, state):
    if state == 0:
      return [0, 1]
    elif state == self.length - 1:
      return [-1, 0]
    else:
      return [-1, 0, 1]
  
  def getTransitionStatesAndProbs(self, state, action):
    state += action
    assert state >= 0 and state < self.length
    return [(state, 1)]
  
  def cost(self, q):
    return 0

  def isTerminal(self, state):
    return state == 0 or state == self.length - 1