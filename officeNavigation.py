import easyDomains
from consQueryAgents import ConsQueryAgent
import time
import random
import numpy
import scipy
import pickle
import gc

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

def classicOfficNav(method):
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
  getBoundedRandLoc = lambda: (random.randint(1, width - 1), random.randint(1, height - 1))

  # specify the size of the domain, which are the robot's possible locations
  width = 5
  height = 5
  # time is 0, 1, ..., horizon
  #horizon = width + height - 1
  
  doors = []#[(width / 2, height / 2)]

  switch = (width - 1, height - 1)
  #switch = getRandLoc()

  numOfCarpets = 5
  carpets = [getBoundedRandLoc() for _ in range(numOfCarpets)]
  
  # number of elements in the query
  k = 2

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
  #walls = [[(width / 2, _), (width / 2 + 1, _)] for _ in range(1, height - 1) if _ != height / 2]
  walls = []
  
  # location, box1, box2, door1, door2, carpet, switch
  allLocations = [(x, y) for x in range(width) for y in range(height)]
  sSets = [allLocations] +\
          [[CLOSED, OPEN] for _ in doors] +\
          [[0, 1]]
  
  aSets = [(1, 0), (0, 1), (1, 1),
           TURNOFFSWITCH]
 
  # factored transition function
  def navigate(s, a):
    loc = s[lIndex]
    if a in [(1, 0), (0, 1), (1, 1)]:
      sp = (loc[0] + a[0], loc[1] + a[1])
      # not blocked by borders, closed doors or walls
      if (sp[0] >= 0 and sp[0] < width and sp[1] >= 0 and sp[1] < height) and\
         not any(s[idx] == CLOSED and sp == doors[idx - dIndexStart] for idx in dIndex) and\
         not any(loc in wall and sp in wall for wall in walls):
        return sp
    return loc
  
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
  # let's not make carpets features but constraints directly
  consStates = [[s for s in mdp['S'] if s[lIndex] == _] for _ in carpets]
  agent = ConsQueryAgent(mdp, consStates)

  relFeats, domPis = agent.findRelevantFeaturesAndDomPis()

  start = time.time()
  if method == 'brute':
    q, mr = agent.findMinimaxRegretConstraintQ(k, relFeats, domPis, pruning=False)
  elif method == 'alg1':
    q, mr = agent.findMinimaxRegretConstraintQ(k, relFeats, domPis, pruning=True)
  elif method == 'chain':
    q, mr = agent.findChaindAdvConstraintQ(k, relFeats, domPis)
  elif method == 'random':
    q, mr = agent.findRandomConstraintQ(k, relFeats, domPis)
  elif method == 'nq':
    q, mr = agent.findNoConstraintQ(k, relFeats, domPis)
  else:
    raise Exception('unknown method', method)
  end = time.time()

  #indices = numpy.random.choice(range(len(relFeats)), len(relFeats) * violableRatio)
  #violableCons = [list(relFeats)[_] for _ in indices]
  
  return mr, end - start


if __name__ == '__main__':
  # default values of parameters
  method = 'alg1'
  mrs = {}
  times = {}

  for method in ['brute', 'alg1', 'chain', 'random', 'nq']:
    for rnd in range(20):
      random.seed(rnd)
      # not necessarily using the following packages, but just to be sure
      numpy.random.seed(rnd)
      scipy.random.seed(rnd)
     
      #flatOfficNav()
      mr, timeElapsed = classicOfficNav(method)

      mrs[(method, rnd)] = mr
      times[(method, rnd)] = timeElapsed
      
      gc.collect()

  pickle.dump(mrs, open('mrs', 'wb'))
  pickle.dump(times, open('times', 'wb'))
