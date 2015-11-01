from cmp import ControlledMarkovProcess

class Sightseeing(ControlledMarkovProcess):
  def __init__(self, queries, trueReward, gamma, responseTime, width, height):
    self.width = width
    self.height = height
    ControlledMarkovProcess.__init__(self, queries, trueReward, gamma, responseTime)

  def cost(self, query):
    return 0

  def getStates(self):
    states = [(x, y, dir, status) for x in xrange(self.width)\
                                  for y in xrange(self.height)\
                                  for dir in [-1, 1]\
                                  for status in xrange(2)]
    states += [(0, y, 0, status) for y in xrange(self.height)\
                                 for status in xrange(2)]
    return states

  def reset(self):
    self.state = (0, 0, 0, 0)
    
  def isTerminal(self, state):
    return state[0] == 0 and state[1] == 0 and state[2] != 0

  def getPossibleActions(self, state):
    # actions are coordinate diff
    if state[2] == 0:
      actions = ['East', 'West']
    else:
      actions = []
      if state[3] != 1: actions.append('Drop')
      
      # can navigate north or south depends on which column
      if state[1] > 0 and state[0] % 2 == 0: actions.append('North')
      if state[1] < self.height - 1 and state[0] % 2 == 1: actions.append('South')
      
      # navigate east or west depending on the direction it chooses
      # not that state[2] == 0 only at the initial state, so it chooses the direction in the first step
      if state[2] <= 0: actions.append('West')
      if state[2] >= 0: actions.append('East')
    
    return actions
  
  def getTransitionStatesAndProbs(self, state, action):
    if action == 'Drop':
      newState = (state[0], state[1], state[2], 1)
    elif action == 'East':
      newState = ((state[0] + 1) % self.width, state[1], 1, 0)
    elif action == 'West':
      newState = ((state[0] - 1) % self.width, state[1], -1, 0)
    elif action == 'South':
      newState = (state[0], state[1] + 1, state[2], 0)
    elif action == 'North':
      newState = (state[0], state[1] - 1, state[2], 0)
    self.stateSanityCheck(newState)
    return [(newState, 1)]
  
  def stateSanityCheck(self, state):
    assert not state[0] < 0
    assert not state[0] >= self.width
    
    assert not state[1] < 0
    assert not state[1] >= self.height
    
    assert state[2] in [-1, 0, 1]
    assert state[3] in [0, 1]