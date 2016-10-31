from cmp import ControlledMarkovProcess

class MountainCar(ControlledMarkovProcess):
  def __init__(self, responseTime, horizon, terminalReward):
    self.noise = 0
    self.acc = 0.1

    self.wallLoc = -5
    self.goal = 10

    # horizon is assumed to be finite in this domain
    ControlledMarkovProcess.__init__(self, responseTime, horizon, terminalReward)
  
  def cost(self, q):
    return 0

  def reset(self):
    return [(0, 0)]
  
  def getStates(self):
    raise Exception('continuous domain')
  
  def getStateDistance(self, s1, s2):
    return abs(s1[0] - s2[0])

  def getPossibleActions(self, state):
    return [-1, 0, 1]

  def isTerminal(self, state):
    return state[0] > self.goal
  
  def getTransitionStatesAndProbs(self, state, action):
    v = state[1]
    loc = state[0] + v

    if action == 1: v += self.acc
    elif action == -1: v -= self.acc
    
    if loc < self.wallLoc: loc = state[0]
     
    return {(loc, v): 1}