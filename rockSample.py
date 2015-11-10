from cmp import ControlledMarkovProcess

class RockSample(ControlledMarkovProcess):
  def __init__(self, queries, trueReward, gamma, responseTime, width, height, horizon, terminalReward):
    self.width = width
    self.height = height
    # horizon is assumed to be finite in this domain
    ControlledMarkovProcess.__init__(self, queries, trueReward, gamma, responseTime, horizon, terminalReward)

  def cost(self, query):
    return 0

  def reset(self):
    # initial state: this far to the first intersection
    self.state = (self.width / 2, 0)
    
  def getStates(self):
    return [(x, y) for x in xrange(self.width) for y in xrange(self.height)]
  
  def getPossibleActions(self, state):
    # actions are coordinate diff
    return [(-1, 0), (1, 0), (0, 1), (0, -1), (0, 0)]
  
  def getTransitionStatesAndProbs(self, state, action):
    newState = self.adjustState((state[0] + action[0], state[1] + action[1]))
    return [(newState, 1)]
  
  def adjustState(self, loc):
    loc = list(loc)

    if loc[0] < 0: loc[0] = 0
    elif loc[0] >= self.width: loc[0] = self.width - 1
    
    if loc[1] < 0: loc[1] = 0
    elif loc[1] >= self.height: loc[1] = self.height - 1
    
    return tuple(loc)