import easyDomains
from consQueryAgents import ConsQueryAgent
import time
import getopt
import sys
import random
import numpy
import scipy
from httplib import MOVED_PERMANENTLY
import util

OPEN = 1
CLOSED = 0

STEPPED = 1
CLEAN = 0

ON = 1
OFF = 0 

INPROCESS = 0
TERMINATED = 1

OPENDOOR = 'openDoor'
CLOSEDOOR = 'closeDoor'
TURNOFFSWITCH = 'turnOffSwitch'
EXIT = 'exit'

def classicOfficNav():
  """
  The office navigation domain specified in the report using a factored representation.
  There are state factors indicating whether some carpets are dirty.
     _________
  2 | |     |S|
  1 |  D_C_D  |
  0 |R___C____|
     0 1 2 3 4
     
  Also, consider randomize the room for experiments.

  FIXME hacking this function too much.
  """
  getRandLoc = lambda: (random.randint(0, width - 2), random.randint(0, height - 2))

  # specify the size of the domain, which are the robot's possible locations
  width = 5
  height = 5
  # time is 0, 1, ..., horizon
  #horizon = width + height - 1
  
  #doors = [(1, 1), (3, 1)]
  doors = []#[(width / 2, height / 2)]

  switch = (width - 1, height - 1)
  #switch = getRandLoc()

  lIndex = 0
  dIndexStart = lIndex + 1
  dSize = len(doors)
  sIndex = dIndexStart + dSize
  # note: time is needed when there are reversible features
  #tIndex = sIndex + 1
  
  dIndex = range(dIndexStart, dIndexStart + dSize)
  
  # pairs of adjacent locations that are blocked by a wall
  #walls = [[(0, 2), (1, 2)], [(1, 0), (1, 1)], [(2, 0), (2, 1)], [(3, 0), (3, 1)], [(3, 2), (4, 2)]]

  # splitting the room into two smaller rooms.
  # the robot can only access to the other room by going through a door in the middle or a corridor at the top
  walls = [[(width / 2, _), (width / 2 + 1, _)] for _ in range(1, height - 1) if _ != height / 2]
  
  # location, box1, box2, door1, door2, carpet, switch
  allLocations = [(x, y) for x in range(width) for y in range(height)]
  sSets = [allLocations] +\
          [[CLOSED, OPEN] for _ in doors] +\
          [[0, 1]]
  
  aSets = [(1, 0), (0, 1), (-1, 0), (0, -1),
           TURNOFFSWITCH]
 
  # factored transition function
  def navigate(s, a):
    loc = s[lIndex]
    if a in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
      sp = (loc[0] + a[0], loc[1] + a[1])
      # not blocked by borders, closed doors or walls
      if (sp[0] >= 0 and sp[0] < width and sp[1] >= 0 and sp[1] < height) and\
         not any(s[idx] == CLOSED and sp == doors[idx - dIndexStart] for idx in dIndex) and\
         not any(loc in wall and sp in wall for wall in walls):
        return sp
    return loc
  
  """
  def carpetOpGen(idx, carpet):
    def carpetOp(s, a):
      loc = s[lIndex]
      carpetState = s[idx]
      if loc == carpet: return STEPPED
      else: return carpetState
    return carpetOp
  """
  
  def doorOpGen(idx, door):
    def doorOp(s, a):
      loc = s[lIndex]
      doorState = s[idx]
      if a in [OPENDOOR, CLOSEDOOR]:
        if loc in [(door[0] - 1, door[1]), (door[0], door[1])]:
          if a == CLOSEDOOR: doorState = CLOSED
          elif a == OPENDOOR: doorState = OPEN
          # otherwise the door state is unchanged
      return doorState
    return doorOp
  
  def switchOp(s, a):
    loc = s[lIndex]
    switchState = s[sIndex]
    if loc == switch and a == 'turnOffSwitch': switchState = OFF 
    return switchState
  
  """
  # time elapses
  def timeOp(s, a):
    return s[-1] + 1
  """

  tFunc = [navigate] +\
          [doorOpGen(dIndexStart + i, doors[i]) for i in range(dSize)] +\
          [switchOp]

  s0List = [(0, 0)] +\
           [CLOSED for _ in doors] +\
           [ON]
  s0 = tuple(s0List)
  
  terminal = lambda s: s[lIndex] == switch
  gamma = .9

  # if need to assign random rewards to all states
  #bonus = util.Counter()
  #for loc in allLocations: bonus[loc] = random.random() < .4

  def reward(s, a):
    if s[lIndex] == switch and s[sIndex] == ON and a == TURNOFFSWITCH:
      return 10
    else:
      # create some random rewards in the domain to break ties
      return 0
  rFunc = reward
  
  mdp = easyDomains.getFactoredMDP(sSets, aSets, rFunc, tFunc, s0, terminal, gamma)

  # states that should not be visited
  numOfCarpets = 5
  consStates = [[s for s in mdp['S'] if s[lIndex] == getRandLoc()] for _ in xrange(numOfCarpets)] # let's not make carpets features but constraints directly
  agent = ConsQueryAgent(mdp, consStates)

  k = 3

  relFeats, domPis = agent.findRelevantFeaturesAndDomPis()

  start = time.time()
  agent.findMinimaxRegretConstraintQ(k, relFeats, domPis, True)
  end = time.time()
  print end - start

  start = time.time()
  agent.findMinimaxRegretConstraintQ(k, relFeats, domPis, False)
  end = time.time()
  print end - start

  #writeToFile('office.out', end - start)

# deleted flat domain here. kinda merged into the function above

def writeToFile(name, value):
  f = open(name, 'a') # not appending
  f.write(str(value) + '\n')
  f.close()


if __name__ == '__main__':
  # default values of parameters
  method = 'alg1'

  try:
    opts, args = getopt.getopt(sys.argv[1:], 'r:a:')
  except getopt.GetoptError:
    raise Exception('Unknown flag')
  for opt, arg in opts:
    if opt == '-r':
      random.seed(int(arg))

      # not necessarily using the following packages, but just to be sure
      numpy.random.seed(int(arg))
      scipy.random.seed(int(arg))
    elif opt == '-a':
      method = arg
 
  #flatOfficNav()
  classicOfficNav()
