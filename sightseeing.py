from cmp import ControlledMarkovProcess

class Sightseeing(ControlledMarkovProcess):
  def __init__(self, queries, trueReward, gamma, responseTime, width, height):
    self.width = width
    self.height = height
    ControlledMarkovProcess.__init__(self, queries, trueReward, gamma, responseTime)

  def cost(self, query):
    return 0

  def reset(self):
    # initial state: this far to the first intersection
    self.state = (0, 0, 0)
    
  def isTerminal(self, state):
    return state[0] == self.width - 1 and state[1] == self.height - 1

  def getStates(self):
    return [(x, y, status) for x in xrange(self.width) for y in xrange(self.height) for status in xrange(2)]
  
  def getPossibleActions(self, state):
    # actions are coordinate diff
    actions = []
    if state[2] != 1: actions.append('Drop')
    if state[0] < self.width - 1: actions.append('East')
    if state[1] < self.height - 1: actions.append('South')
    
    return actions
  
  def getTransitionStatesAndProbs(self, state, action):
    if action == 'Drop':
      newState = (state[0], state[1], 1)
    elif action == 'East':
      newState = (state[0] + 1, state[1], 0)
    elif action == 'South':
      newState = (state[0], state[1] + 1, 0)
    self.stateSanityCheck(newState)
    return [(newState, 1)]
  
  def stateSanityCheck(self, state):
    assert not state[0] < 0
    assert not state[0] >= self.width
    
    assert not state[1] < 0
    assert not state[1] >= self.height