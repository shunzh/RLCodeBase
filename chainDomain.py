from cmp import ControlledMarkovProcess

class ChainDomain(ControlledMarkovProcess):
  def __init__(self, rewardFunc, queries, length=5):
    responseFunc = lambda state: rewardFunc(state)
    self.length = length
    self.responseTime = 1

    ControlledMarkovProcess.__init__(self, queries, responseFunc)
  
  def getStates(self):
    return range(self.length)
  
  def getStartState(self):
    return self.length / 2
  
  def getPossibleActions(self, state):
    if state == 0:
      return [0, 1]
    elif state == self.length - 1:
      return [-1, 0]
    else:
      return [-1, 0, 1]
  
  def getTransitionStatesAndProbs(self, state, action):
    state += action
    if state < 0: state = 0
    elif state >= self.length: state = self.length - 1
    return [(state, 1)]
  
  def cost(self, q):
    return 0

  def isTerminal(self, state):
    return state == 0 or state == self.length - 1