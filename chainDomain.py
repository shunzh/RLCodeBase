from cmp import ControlledMarkovProcess

class ChainDomain(ControlledMarkovProcess):
  def __init__(self, rewardFunc, length=5):
    responseFunc = lambda state: rewardFunc(state)
    ControlledMarkovProcess.__init__(self, responseFunc)
    
    self.length = length
  
  def getStates(self):
    return range(self.length)
  
  def getStartState(self):
    return self.length / 2
  
  def getPossibleActions(self, state):
    return [-1, 1]
  
  def getTransitionStatesAndProbs(self, state, action):
    nextState = state + action
    if state < 0: state = 0
    elif state >= self.length: state = self.length - 1
  
  def isTerminal(self, state):
    return False