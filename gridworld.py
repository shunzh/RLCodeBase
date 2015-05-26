# gridworld.py
# ------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import mdp
import environment
import util

class Gridworld(mdp.MarkovDecisionProcess):
  """
    Gridworld
  """
  def __init__(self, grid, isFinal):
    # layout
    if type(grid) == type([]): grid = makeGrid(grid)
    self.grid = grid
    
    # parameters
    self.livingReward = 0.0
    self.noise = 0.0
    self.isFinal = isFinal
        
  def setLivingReward(self, reward):
    """
    The (negative) reward for exiting "normal" states.
    
    Note that in the R+N text, this reward is on entering
    a state and therefore is not clearly part of the state's
    future rewards.
    """
    self.livingReward = reward
        
  def setNoise(self, noise):
    """
    The probability of moving in an unintended direction.
    """
    self.noise = noise
                                    
  def getPossibleActions(self, state):
    """
    Returns list of valid actions for 'state'.
    
    Note that you can request moves into walls and
    that "exit" states transition to the terminal
    state under the special action "done".
    """
    #return ('north','west','south','east', 'ne', 'se', 'nw', 'sw')
    return ('north','west','south','east')
    
  def getStates(self):
    """
    Return list of all states.
    """
    # The true terminal state.
    states = []
    for x in range(self.grid.width):
      for y in range(self.grid.height):
        if self.grid[x][y] != '#':
          state = (x,y)
          states.append(state)
    return states
        
  def getReward(self, state, action, nextState):
    """
    Get reward for state, action, nextState transition.
    
    Note that the reward depends only on the state being
    departed (as in the R+N book examples, which more or
    less use this convention).
    """
    if nextState == self.grid.terminalState:
      return 0.0
    x, y = nextState
    cell = self.grid[x][y]
    if type(cell) == int or type(cell) == float:
      return cell
    return self.livingReward
        
  def setReward(self, state, reward):
    if state != self.grid.terminalState:
      x, y = state
      self.grid[x][y] = reward
      
  def getStartState(self):
    startStates = []
    for x in range(self.grid.width):
      for y in range(self.grid.height):
        if self.grid[x][y] == 'S':
          startStates.append((x, y))

    if len(startStates) == 0:
      raise 'Grid has no start state'
    else:
      return random.choice(startStates)  
    
  def isTerminal(self, state):
    """
    Only the TERMINAL_STATE state is *actually* a terminal state.
    The other "exit" states are technically non-terminals with
    a single action "exit" which leads to the true terminal state.
    This convention is to make the grids line up with the examples
    in the R+N textbook.
    """
    return self.isFinal(state)
                   
  def getTransitionStatesAndProbs(self, state, action):
    """
    Returns list of (nextState, prob) pairs
    representing the states reachable
    from 'state' by taking 'action' along
    with their transition probabilities.          
    """        
    if action not in self.getPossibleActions(state):
      raise Exception("Illegal action " + str(action))
      
    if action == 'exit':
      return [(self.grid.terminalState, 1)]

    if self.isTerminal(state):
      return []
    
    x, y = state
    
    successors = []                
                
    northState = (self.__isAllowed(y+1,x) and (x,y+1)) or state
    westState = (self.__isAllowed(y,x-1) and (x-1,y)) or state
    southState = (self.__isAllowed(y-1,x) and (x,y-1)) or state
    eastState = (self.__isAllowed(y,x+1) and (x+1,y)) or state

    if action == 'north' or action == 'south':
      if action == 'north': 
        successors.append((northState,1-self.noise))
      else:
        successors.append((southState,1-self.noise))
                                
      massLeft = self.noise
      successors.append((westState,massLeft/2.0))    
      successors.append((eastState,massLeft/2.0))
                                
    if action == 'west' or action == 'east':
      if action == 'west':
        successors.append((westState,1-self.noise))
      else:
        successors.append((eastState,1-self.noise))
                
      massLeft = self.noise
      successors.append((northState,massLeft/2.0))
      successors.append((southState,massLeft/2.0)) 
      
    successors = self.__aggregate(successors)

    return successors                                
  
  def __aggregate(self, statesAndProbs):
    counter = util.Counter()
    for state, prob in statesAndProbs:
      counter[state] += prob
    newStatesAndProbs = []
    for state, prob in counter.items():
      newStatesAndProbs.append((state, prob))
    return newStatesAndProbs
        
  def __isAllowed(self, y, x):
    if y < 0 or y >= self.grid.height: return False
    if x < 0 or x >= self.grid.width: return False
    return self.grid[x][y] != '#'

class GridworldEnvironment(environment.Environment):
    
  def __init__(self, gridWorld):
    self.gridWorld = gridWorld
    self.reset()
            
  def getCurrentState(self):
    return self.state
        
  def getPossibleActions(self, state):        
    return self.gridWorld.getPossibleActions(state)
        
  def doAction(self, action):
    successors = self.gridWorld.getTransitionStatesAndProbs(self.state, action) 
    sum = 0.0
    rand = random.random()
    state = self.getCurrentState()
    for nextState, prob in successors:
      sum += prob
      if sum > 1.0:
        raise 'Total transition probability more than one; sample failure.' 
      if rand < sum:
        reward = self.gridWorld.getReward(state, action, nextState)
        self.state = nextState
        return (nextState, reward)
    raise 'Total transition probability less than one; sample failure.'    
        
  def reset(self):
    self.state = self.gridWorld.getStartState()

  def isFinal(self):
    return self.gridWorld.isFinal(self.state)

class Grid:
  """
  A 2-dimensional array of immutables backed by a list of lists.  Data is accessed
  via grid[x][y] where (x,y) are cartesian coordinates with x horizontal,
  y vertical and the origin (0,0) in the bottom left corner.  
  
  The __str__ method constructs an output that is oriented appropriately.
  """
  def __init__(self, width, height, initialValue=' '):
    self.width = width
    self.height = height
    self.data = [[initialValue for y in range(height)] for x in range(width)]
    self.terminalState = 'TERMINAL_STATE'
    
  def __getitem__(self, i):
    return self.data[i]
  
  def __setitem__(self, key, item):
    self.data[key] = item
    
  def __eq__(self, other):
    if other == None: return False
    return self.data == other.data
    
  def __hash__(self):
    return hash(self.data)
  
  def copy(self):
    g = Grid(self.width, self.height)
    g.data = [x[:] for x in self.data]
    return g
  
  def deepCopy(self):
    return self.copy()
  
  def shallowCopy(self):
    g = Grid(self.width, self.height)
    g.data = self.data
    return g
    
  def _getLegacyText(self):
    t = [[self.data[x][y] for x in range(self.width)] for y in range(self.height)]
    t.reverse()
    return t
    
  def __str__(self):
    return str(self._getLegacyText())

def makeGrid(gridString):
  width, height = len(gridString[0]), len(gridString)
  grid = Grid(width, height)
  for ybar, line in enumerate(gridString):
    #y = height - ybar - 1
    y = ybar
    for x, el in enumerate(line):
      grid[x][y] = el
  return grid    
             
def terminateIfInt(grid):
	return lambda state : type(grid[state[1]][state[0]]) == int

def getDiscountedRewardPathGrid():
  grid = [[.9 ** dist] * 5 for dist in reversed(range(5))]
  grid[0][0] = 'S'
  isFinal = lambda state: state[1] == 4
  return Gridworld(makeGrid(grid), isFinal)

def getCliffGrid():
  grid = [[' ',' ',' ',' ',' '],
          ['S',' ',' ',' ',10],
          [-100,-100, -100, -100, -100]]
  return Gridworld(makeGrid(grid), terminateIfInt(grid))
    
def getCliffGrid2():
  grid = [[' ',' ',' ',' ',' '],
          [8,'S',' ',' ',10],
          [-100,-100, -100, -100, -100]]
  return Gridworld(grid, terminateIfInt(grid))
    
def getDiscountGrid():
  grid = [[' ',' ',' ',' ',' '],
          [' ','#',' ',' ',' '],
          [' ','#', 1,'#', 10],
          ['S',' ',' ',' ',' '],
          [-10,-10, -10, -10, -10]]
  return Gridworld(grid, terminateIfInt(grid))
   
def getBridgeGrid():
  grid = [[ '#',-100, -100, -100, -100, -100, '#'],
          [   1, 'S',  ' ',  ' ',  ' ',  ' ',  10],
          [ '#',-100, -100, -100, -100, -100, '#']]
  return Gridworld(grid, terminateIfInt(grid))

def getBookGrid():
  grid = [[' ',' ',' ',+1],
          [' ','#',' ',-1],
          ['S',' ',' ',' ']]
  return Gridworld(grid, terminateIfInt(grid))

def getMazeGrid():
  grid = [[' ',' ',' ',+1],
          ['#','#',' ','#'],
          [' ','#',' ',' '],
          [' ','#','#',' '],
          ['S',' ',' ',' ']]
  return Gridworld(grid, terminateIfInt(grid))

def getObstacleGrid():
  grid = [[' ',' ',' ',' ',' '],
          [' ','S','S','S',' '],
          [' ','S', -1,'S',' '],
          [' ','S','S','S',' '],
          [' ',' ',' ',' ',' ']]
  isFinal = lambda state : False
  return Gridworld(grid, isFinal)

def getSidewalkGrid():
  grid = [[ 'S',' ', ' ', ' ', +1],
          [ 'S',' ', ' ', ' ', +1],
          [ 'S',' ', ' ', ' ', +1]]
  isFinal = lambda state : state[0] == 4
  return Gridworld(grid, isFinal)

def getWalkAvoidGrid():
  grid = [[ 'S',' ', ' ',  +1, ' ',  +1, +2],
          [ 'S',' ', ' ', ' ', ' ', ' ', +2],
          [ 'S',' ',  -1, ' ',  -1, ' ', +2]]
  isFinal = lambda state : state[0] == 6
  return Gridworld(grid, isFinal)

def getToyWalkAvoidGrid():
  grid = [[ 'S', ' '],
          [ ' ', 1]]
  isFinal = lambda state : state[0] == 1 and state[1] == 1
  return Gridworld(grid, isFinal)
 
def getLargeWalkAvoidGrid(obstacleProportion = 0.2):
  """
    Randomly generate a large grid
  """
  width = 10
  height = 10
  targetProportion = 0.05

  # init grid world
  grid = [[' ' for i in range(width)] for j in range(height)]

  # add start and end states
  for j in range(height):
    grid[j][0] = 'S'
    grid[j][width - 1] = +2

  # random generator used in this context
  rand = random.Random()
  rand.seed(0)

  # randomly set obstacles
  for _ in xrange(int(width * height * obstacleProportion)):
    while True:
      y = rand.choice(range(height))
      x = rand.choice(range(1, width - 1))
      if grid[y][x] == ' ':
        grid[y][x] = -1
        break

  # randomly set targets
  for _ in xrange(int(width * height * targetProportion)):
    while True:
      y = rand.choice(range(height))
      x = rand.choice(range(1, width - 1))
      if grid[y][x] == ' ':
        grid[y][x] = +1
        break

  isFinal = lambda state : state[0] == width - 1
  return Gridworld(grid, isFinal)

