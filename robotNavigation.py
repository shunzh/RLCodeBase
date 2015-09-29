from cmp import ControlledMarkovProcess

class RobotNavigation(ControlledMarkovProcess):
  def __init__(self, queries, trueReward, gamma, responseTime, width, height):
    self.width = width
    self.height = height
    ControlledMarkovProcess.__init__(self, queries, trueReward, gamma, responseTime)

  def cost(self, query):
    return 0

  def reset(self):
    # initial state: this far to the first intersection
    self.state = (0, self.height / 2)
    
  def getStates(self):
    return [(x, y) for x in xrange(self.width) for y in xrange(self.height)]
  
  def getPossibleActions(self, state):
    # actions are coordinate diff
    return [(-1, 0), (1, 0), (0, 1), (0, -1), (0, 0)]
  
  def getTransitionStatesAndProbs(self, state, action):
    newState = self.stateSanityCheck((state[0] + action[0], state[1] + action[1]))
    return [(newState, 1)]
  
  def stateSanityCheck(self, state):
    state = list(state)

    if state[0] < 0: state[0] = 0
    elif state[0] >= self.width: state[0] = self.width - 1
    
    if state[1] < 0: state[1] = 0
    elif state[1] >= self.height: state[1] = self.height - 1
    
    return tuple(state)