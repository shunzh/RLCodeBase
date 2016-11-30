from cmp import ControlledMarkovProcess

class MountainCar(ControlledMarkovProcess):
  def __init__(self, responseTime, horizon, terminalReward):
    self.noise = 0
    self.acc = 0.1

    self.wallLoc = -5
    self.goal = 5

    # horizon is assumed to be finite in this domain
    ControlledMarkovProcess.__init__(self, responseTime, horizon, terminalReward)
  
  def cost(self, q):
    return 0

  def reset(self):
    self.state = (0, 0)
  
  def getStates(self):
    #FIXME should not be used
    return []
  
  def getStateDistance(self, s1, s2):
    return abs(s1[0] - s2[0])

  def getPossibleActions(self, state = None):
    # 0: stay the same speed
    # -1, 1: accelerate forward or backward
    return [-1, 0, 1]

  def isTerminal(self, state):
    return state[0] > self.goal or state[0] < self.wallLoc
  
  def getTransitionStatesAndProbs(self, state, action):
    v = state[1]

    if action == 1: v += self.acc
    elif action == -1: v -= self.acc
    
    # -2 <= v <= 2
    v = max(min(v, 1), -1)

    loc = state[0] + v
     
    return [((loc, v), 1)]


class MountainCarToy(MountainCar):
  """
  for testing.. the agent controls velocity not acceleration
  """
  def getTransitionStatesAndProbs(self, state, action):
    loc = state[0] + action
    
    return [((loc, 0), 1)]