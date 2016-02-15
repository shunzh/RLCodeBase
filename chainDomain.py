from cmp import ControlledMarkovProcess
import numpy as np

class ChainDomain(ControlledMarkovProcess):
  def __init__(self, queries, trueReward, gamma, responseTime):
    ControlledMarkovProcess.__init__(self, queries, trueReward, gamma, responseTime, horizon=np.inf, terminalReward=None)
  
  def getStates(self):
    states = []
    states += [(0, i) for i in range(4)]
    states += [(l, i) for l in range(2, 4) for i in range(2)]
    states += [(l, 0) for l in range(4, 8)]
    return states
  
  def getStartState(self):
    return (0, 0)
  
  def getPossibleActions(self, state):
    if state[0] >= 2 and state[0] < 4 and state[1] == 1:
      return [0, 1]
    else:
      return [0]
  
  def getTransitionStatesAndProbs(self, state, action):
    state = (state[0], state[1] + 1)
    if state[0] == 0 and state[1] == 4:
      return [((2, 0), .5), ((3, 0), .5)]
    elif state[0] >= 2 and state[0] < 4 and state[1] == 2:
      return [((state[0] * 2 + action, 0), 1)]
    else:
      return [(state, 1)]
  
  def cost(self, q):
    return 0

  def isTerminal(self, state):
    return state[0] >= 4