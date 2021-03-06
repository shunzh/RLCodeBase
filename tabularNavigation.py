from cmp import ControlledMarkovProcess
import util
import random
import config

class TabularNavigation(ControlledMarkovProcess):
  def __init__(self, responseTime, width, height, horizon, terminalReward):
    self.width = width
    self.height = height
    self.noise = 0.05
    # horizon is assumed to be finite in this domain
    ControlledMarkovProcess.__init__(self, responseTime, horizon, terminalReward)

  def cost(self, query):
    return 0

  def reset(self):
    # initial state: this far to the first intersection
    self.state = (0, 0)
    
  def getStates(self):
    return [(x, y) for x in xrange(self.width) for y in xrange(self.height)]
  
  def getPossibleActions(self, state=None):
    # actions are coordinate diff
    return [(1, 0), (0, 1), (-1, 0), (0, -1)]

  def isTerminal(self, state):
    return state[0] == self.width - 1 and state[1] == self.height - 1
 
  def getStateDistance(self, s1, s2):
    return abs(s1[0] - s2[0]) + abs(s1[1] - s2[1])

  def getTransitionStatesAndProbs(self, state, action):
    if self.isTerminal(state): return []
    else:
      transProb = {self.adjustState((state[0] + a[0], state[1] + a[1])): self.noise for a in self.getPossibleActions()}
      transSize = len(transProb.keys())

      newState = self.adjustState((state[0] + action[0], state[1] + action[1]))
      transProb[newState] = 1 - self.noise * (transSize - 1)

      return transProb.items()
  
  def adjustState(self, loc):
    loc = list(loc)

    if loc[0] < 0: loc[0] = 0
    elif loc[0] >= self.width: loc[0] = self.width - 1
    
    if loc[1] < 0: loc[1] = 0
    elif loc[1] >= self.height: loc[1] = self.height - 1
    
    return tuple(loc)

  def measure(self, state1, state2):
    return abs(state1[0] - state2[0]) + abs(state1[1] - state2[1])


class ThreeStateToy(TabularNavigation):
  def __init__(self, responseTime, horizon, terminalReward):
    # no need to make this domain noisy
    self.noise = 0
    ControlledMarkovProcess.__init__(self, responseTime, horizon, terminalReward)

    self.feats = util.Counter()
    for s in self.getStates():
      for a in self.getPossibleActions():
        self.feats[(s, a)] = [0]
    self.feats[(0, 0), (-1, 0)] = [-1]
    self.feats[(0, 0), (1, 0)] = [1]
    self.feats[(0, 0), (0, 1)] = [0]

  def getStates(self):
    return [(0, 0), (-1, 0), (1, 0), (0, 1)]
  
  def reset(self):
    self.state = (0, 0)
  
  def getPossibleActions(self, state=None):
    return [(-1, 0), (1, 0), (0, 1)]
  
  def isTerminal(self, state):
    return not state == (0, 0)
  
  def adjustState(self, loc):
    # no need
    return loc


class RockCollection(TabularNavigation):
  def __init__(self, responseTime, width, height, horizon, terminalReward, rockNum):
    """
    args:
      rockNum: number of rocks to have potentially nontrivial features
      dimension: dimensionality of the features
    """
    TabularNavigation.__init__(self, responseTime, width, height, horizon, terminalReward)
    
    # initialize the features for rocks
    # assume each feature for rocks can be either 1 or 0
    self.feats = util.Counter()
    rocks = random.sample(filter(lambda (x, y): y < height - 1, self.getStates()), rockNum)
    # we assign features randomly for the rocks
    # let different actions share the same feature
    for state in self.getStates():
      for action in self.getPossibleActions():
        if state in rocks:
          self.feats[state, action] = [random.choice([0, 1]) for _ in range(config.DIMENSION)]
        else:
          self.feats[state, action] = [0,] * config.DIMENSION
  
  def getPossibleActions(self, state=None):
    return [(-1, 1), (0, 1), (1, 1)]

  def isTerminal(self, state):
    return state[1] == self.height - 1


class TabularNavigationKWay(TabularNavigation):
  degree = 3

  def __init__(self, responseTime, width, height, horizon, terminalReward):
    # pre-build transitions 
    self.transit = util.Counter()
    for y in xrange(height - 1):
      for x in xrange(self.getNumOfStatesPerRow(y)):
        if config.CONNECTION_TYPE == 'tree':
          self.transit[(x, y)] = [(x * self.degree + i, y + 1) for i in range(self.degree)]
        elif config.CONNECTION_TYPE == 'grid':
          self.transit[(x, y)] = [(x + i, y + 1) for i in range(self.degree)]
        elif config.CONNECTION_TYPE == 'chain':
          self.transit[(x, y)] = [(x, y + 1)] * self.degree
        else:
          raise Exception('Unknown connection type of k way navigation.')

    TabularNavigation.__init__(self, responseTime, width, height, horizon, terminalReward)

  @staticmethod
  def getNumOfStatesPerRow(y):
    if config.CONNECTION_TYPE == 'tree':
      return TabularNavigationKWay.degree ** y
    elif config.CONNECTION_TYPE == 'grid':
      return 1 + y * (TabularNavigationKWay.degree - 1) 
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
      transProb = {self.transit[state][a]: self.noise for a in self.getPossibleActions()}
      transProb[self.transit[state][action]] = 1 - self.noise * (self.degree - 1)
      return transProb.items()
  
  def isTerminal(self, state):
    return state[1] == self.height - 1
 

class TabularNavigationMaze(TabularNavigation):
  """
  Split the world into 4 rooms
  """
  def __init__(self, queries, trueReward, gamma, responseTime, width, height, horizon, terminalReward):
    self.walls = [(x, height / 2) for x in range(width / 2 - 1) + range(width / 2 + 2, width)]
    self.walls += [(width / 2, y) for y in range(height / 2 - 1) + range(height / 2 + 2, height)]
    TabularNavigation.__init__(self, queries, trueReward, gamma, responseTime, width, height, horizon, terminalReward)

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


class DiscreteDriving(RockCollection):
  def __init__(self, numOfCars, responseTime, width, height, horizon, terminalReward):
    TabularNavigation.__init__(self, responseTime, width, height, horizon, terminalReward)
    self.numOfCars = numOfCars
    self.noise = 0
    
    # initialize cars
    self.cars = []
    for _ in range(numOfCars):
      x = random.randint(0, width - 1)
      y = random.randint(0, height - 2)
      self.cars.append((x, y))
    #self.cars = [(2, 3)] #FIXME for testing


class PuddleWorld(TabularNavigation):
  def reset(self):
    self.state = (self.width / 2, 0)
  
  def getPossibleActions(self, state):
    return [(-1, 1), (0, 1), (1, 1)]

  def isTerminal(self, state):
    return state[1] == self.height - 1

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
