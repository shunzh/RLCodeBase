from cmp import ControlledMarkovProcess
import random

class Driving(ControlledMarkovProcess):
  def __init__(self, responseTime, horizon, terminalReward, length=10, lanes=5):
    self.noise = 0
    
    self.lanes = lanes 
    self.length = length
    
    # horizon is assumed to be finite in this domain
    ControlledMarkovProcess.__init__(self, responseTime, horizon, terminalReward)
    
  def cost(self, q):
    return 0
  
  def reset(self):
    self.state = (0, self.lanes / 2)
    
  def getStates(self):
    return []
  
  def sampleState(self):
    loc = random.random() * self.length
    lane = random.randint(0, self.lanes - 1)
    return (loc, lane)
  
  def getStateDistance(self, s1, s2):
    return abs(s1[0] - s2[0]) + abs(s1[1] - s2[1])
  
  def getPossibleActions(self, state=None):
    return ['W', 'E', 'N']
  
  def isTerminal(self, state):
    # only depends on the task horizon
    return state[0] >= self.length
  
  def getTransitionStatesAndProbs(self, state, action):
    loc = state[0] + 0.1
    
    if action == 'W':
      lane = max(state[1] - 1, 0)
    elif action == 'E':
      lane = min(state[1] + 1, self.lanes - 1)
    else:
      lane = state[1]
      
    return [((loc, lane), 1)]
