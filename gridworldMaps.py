from gridworld import Gridworld
import random

def terminateIfInt(grid):
  return lambda state : type(grid[state[1]][state[0]]) == int

def getDiscountedRewardPathGrid():
  grid = [[.9 ** dist] * 5 for dist in reversed(range(5))]
  grid[0][0] = 'S'
  isFinal = lambda state: state[1] == 4
  return Gridworld(grid, isFinal)

def getCliffGrid():
  grid = [[' ',' ',' ',' ',' '],
          ['S',' ',' ',' ',10],
          [-100,-100, -100, -100, -100]]
  return Gridworld(grid, terminateIfInt(grid))
    
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
  isFinal = lambda state : False
  return Gridworld(grid, isFinal)

def getLargeWalkAvoidGrid(width, height, specification):
  """
    Randomly generate a large grid
  """
  # make sure we have enough space for all the objects
  assert width * height - 2 >= sum([count for reward, count in specification])

  # init grid world
  grid = [[' ' for i in range(width)] for j in range(height)]

  # add start and end states
  for j in range(height):
    grid[j][0] = 'S'
    grid[j][width - 1] = 0

  # random generator used in this context
  rand = random.Random()
  rand.seed(0)

  for reward, count in specification:
    # randomly set objects
    for _ in xrange(count):
      while True:
        y = rand.choice(range(height))
        x = rand.choice(range(1, width - 1))
        if grid[y][x] == ' ':
          grid[y][x] = reward
          break

  isFinal = lambda state : state[0] == width - 1
  mdp = Gridworld(grid, isFinal)
  mdp.spec = specification
  
  return mdp

# a specification used for nips paper
getRuohanGrid = lambda: getLargeWalkAvoidGrid(20, 20, [(1, 10), (2, 10), (-1, 10), (-2, 10)])