from cmp import ControlledMarkovProcess
from __builtin__ import True

class NWayCorridor(ControlledMarkovProcess):
  def __init__(self, n, queries, trueReward, gamma, responseTime, horizon, terminalReward):
    ControlledMarkovProcess.__init__(self, queries, trueReward, gamma, responseTime, horizon, terminalReward)
    self.n = n
    self.corridorLen = 5
  
  def cost(self, query):
    return 0
  
  def reset(self):
    self.state = (0, 0)
  
  def getStates(self):
    states = [(0, 0)]
    states += [(x, y) for x in xrange(self.n) for y in xrange(self.corridorLen)]
    
    return states
  
  def getPossibleActions(self, state):
    if state == (0, 0):
      return range(self.n)
    else:
      return ["Move"]
  
  def getTransitionStatesAndProbs(self, state, action):
    if action in range(self.n):
      nextState = [action, 0]
    elif action == 'Move':
      nextState = [state[0], state[1] + 1]
    
    return [(nextState, 1)]
  
  def isTerminal(self, state):
    if state[1] == self.corridorLen - 1: return True
    else: return False