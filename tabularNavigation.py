from cmp import ControlledMarkovProcess
import util
import random
import config
import numpy

class TabularNavigation(ControlledMarkovProcess):
  def __init__(self, queries, trueReward, gamma, responseTime, width, height, horizon, terminalReward):
    self.width = width
    self.height = height
    self.noise = 0.1
    # horizon is assumed to be finite in this domain
    ControlledMarkovProcess.__init__(self, queries, trueReward, gamma, responseTime, horizon, terminalReward)

  def cost(self, query):
    return 0

  def reset(self):
    # initial state: this far to the first intersection
    self.state = (0, 0)
    
  def getStates(self):
    return [(x, y) for x in xrange(self.width) for y in xrange(self.height)]
  
  def getPossibleActions(self, state=None):
    # actions are coordinate diff
    return [(1, 0), (0, 1)]

  def isTerminal(self, state):
    return state == (self.width - 1, self.height - 1)
 
  def getTransitionStatesAndProbs(self, state, action):
    if self.isTerminal(state): return []
    elif state[0] == self.width - 1: return [((state[0], state[1] + 1), 1)]
    elif state[1] == self.height - 1: return [((state[0] + 1, state[1]), 1)]
    else:
      possibleNewStates = map(lambda a: (state[0] + a[0], state[1] + a[1]), self.getPossibleActions(state))
      sap = {s: self.noise for s in possibleNewStates}

      newState = (state[0] + action[0], state[1] + action[1])
      sap[newState] = 1 - self.noise * (len(self.getPossibleActions()) - 1)

      return sap.items()
  
  def adjustState(self, loc):
    loc = list(loc)

    if loc[0] < 0: loc[0] = 0
    elif loc[0] >= self.width: loc[0] = self.width - 1
    
    if loc[1] < 0: loc[1] = 0
    elif loc[1] >= self.height: loc[1] = self.height - 1
    
    return tuple(loc)

  def measure(self, state1, state2):
    return abs(state1[0] - state2[0]) + abs(state1[1] - state2[1])


class TabularNavigationKWay(TabularNavigation):
  def __init__(self, queries, trueReward, gamma, responseTime, width, height, horizon, terminalReward):
    self.degree = config.para

    # pre-build transitions 
    self.transit = util.Counter()
    for y in xrange(height - 1):
      for x in xrange(self.getNumOfStatesPerRow(y)):
        if config.CONNECTION_TYPE == 'tree':
          self.transit[(x, y)] = [(2 * x, y + 1), (2 * x + 1, y + 1)]
        elif config.CONNECTION_TYPE == 'grid':
          self.transit[(x, y)] = [(x, y + 1), (x + 1, y + 1)]
        elif config.CONNECTION_TYPE == 'chain':
          self.transit[(x, y)] = [(x, y + 1), (x, y + 1)]
        else:
          raise Exception('Unknown connection type of k way navigation.')

    TabularNavigation.__init__(self, queries, trueReward, gamma, responseTime, width, height, horizon, terminalReward)

  @staticmethod
  def getNumOfStatesPerRow(y):
    if config.CONNECTION_TYPE == 'tree':
      return config.para ** y
    elif config.CONNECTION_TYPE == 'grid':
      return y + 1
    elif config.CONNECTION_TYPE == 'chain':
      return 1
    else:
      raise Exception('Unknown connection type of k way navigation.')
    
  def getStates(self):
    ret = []
    for y in xrange(self.height):
      ret += [(x, y) for x in xrange(self.getNumOfStatesPerRow(y))]
    return ret
    
  def getPossibleActions(self, state=None):
    return range(self.degree)
  
  def getTransitionStatesAndProbs(self, state, action):
    if self.isTerminal(state): return []
    else:
      # we already computed the transition function in init
      return [(self.transit[state][action], 1)]
  
  def isTerminal(self, state):
    return state[1] == self.height - 1
 

class TabularNavigationToy(TabularNavigation):
  def reset(self):
    # initial state: this far to the first intersection
    self.state = (0, 0)
 
  def isTerminal(self, state):
    return state in [(1, 0), (1, 1), (0, 2), (1, 2)]


class TabularNavigationMaze(TabularNavigation):
  """
  Split the world into 4 rooms
  """
  def __init__(self, queries, trueReward, gamma, responseTime, width, height, horizon, terminalReward):
    self.walls = [(x, height / 2) for x in range(width / 2 - 1) + range(width / 2 + 2, width)]
    self.walls += [(width / 2, y) for y in range(height / 2 - 1) + range(height / 2 + 2, height)]
    TabularNavigation.__init__(self, queries, trueReward, gamma, responseTime, width, height, horizon, terminalReward)

  def getTransitionStatesAndProbs(self, state, action):
    newState = TabularNavigation.getTransitionStatesAndProbs(self, state, action)[0][0]
    if newState in self.walls:
      # bump into walls
      newState = state
    
    return [(newState, 1)]

  def measure(self, state1, state2):
    # mind the obstacles
    #FXIME overfit the current map
    mid = (self.width / 2, self.height / 2)
    if (state1[0] - mid[0]) * (state2[0] - mid[0]) > 0 and\
       (state1[1] - mid[1]) * (state2[1] - mid[1]) > 0:
      return TabularNavigation.measure(self, state1, state2)
    else:
      return TabularNavigation.measure(self, state1, mid)\
           + TabularNavigation.measure(self, state2, mid)


class PuddleWorld(TabularNavigation):
  def reset(self):
    self.state = (self.width / 2, 0)
  
  def getPossibleActions(self, state):
    return [(-1, 1), (0, 1), (1, 1)]

  def getReachability(self):
    xmax = util.Counter()
    xmin = util.Counter()
    xmin[self.state] = 1
    xmax[self.state] = 1
    
    # compute reachability
    for y in range(1, self.height):
      for x in range(self.width):
        xmax[(x, y)] = max([xmax[(prevX, y - 1)] for prevX in [x-1, x, x+1]])

    # compute commitment-satisfying condition
    for x in range(self.width):
      if self.terminalReward[(x, self.height - 1)] < 0:
        xmax[(x, self.height - 1)] = 0

    for y in reversed(range(self.height - 1)):
      for x in range(self.width):
        if max([xmax[(nextX, y + 1)] for nextX in [x-1, x, x+1]]) == 0:
          xmax[(x, y)] = 0

    """
    for y in reversed(range(self.height - 1)):
      for x in range(self.width):
        print xmax[(x, y)],
      print 
    """
    return xmin, xmax
