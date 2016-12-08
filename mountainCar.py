from cmp import ControlledMarkovProcess
import math
import random

class MountainCar(ControlledMarkovProcess):
  def __init__(self, responseTime, horizon, terminalReward):
    self.noise = 0
    
    self.acc = 0.01

    self.wallLoc = -1.2
    self.goal = 1.2

    # horizon is assumed to be finite in this domain
    ControlledMarkovProcess.__init__(self, responseTime, horizon, terminalReward)
  
  def cost(self, q):
    return 0

  def reset(self):
    self.state = (0, 0)
  
  def getStates(self):
    #FIXME should not be used
    return []
  
  def sampleState(self):
    x = self.wallLoc + (self.goal - self.wallLoc) * random.random()
    v = -1 + 2 * random.random()
    return (x, v)
    
  def getStateDistance(self, s1, s2):
    return abs(s1[0] - s2[0])

  def getPossibleActions(self, state = None):
    # 0: stay the same speed
    # -1, 1: accelerate forward or backward
    return [-1, 0, 1]

  def isTerminal(self, state):
    return state[0] > self.goal or state[0] < self.wallLoc
  
  def getTransitionStatesAndProbs(self, state, action):
    x, v = state

    x = x + v
    v = v + action * self.acc

    v = max(min(v, 1), -1)
     
    return [((x, v), 1)]


class RealMountainCar(MountainCar):
  def __init__(self, **args):
    MountainCar.__init__(self, **args)
    
    self.wallLoc = -1.2
    self.goal = 0.6

  def getTransitionStatesAndProbs(self, state, action):
    x, v = state
    x = x + v
    v = v + 0.001 * action - 0.0025 * math.cos(3 * x)
    
    # range of variables
    x = max(min(x, 0.6), -1.2)
    v = max(min(v, 0.07), -0.07)
     
    return [((x, v), 1)]
  
  def isTerminal(self, state):
    return state[0] > self.goal


class MountainCarToy(MountainCar):
  """
  for testing.. the agent controls velocity not acceleration
  """
  def getTransitionStatesAndProbs(self, state, action):
    loc = state[0] + action
    
    return [((loc, 0), 1)]
