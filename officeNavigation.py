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
  horizon = width + height - 1
  
  # some objects
  #carpets = [(2, 0), (2, 1)]
  carpets = [getRandLoc() for _ in xrange(5)]

  #doors = [(1, 1), (3, 1)]
  doors = [(width / 2, height / 2)]

  switch = (width - 1, height - 1)
  #switch = getRandLoc()

  lIndex = 0
  cIndexStart = lIndex + 1
  cSize = len(carpets)
  dIndexStart = cIndexStart + cSize
  dSize = len(doors)
  sIndex = dIndexStart + dSize
  tIndex = sIndex + 1
  
  cIndex = range(cIndexStart, cIndexStart + cSize)
  dIndex = range(dIndexStart, dIndexStart + dSize)
  
  # pairs of adjacent locations that are blocked by a wall
  #walls = [[(0, 2), (1, 2)], [(1, 0), (1, 1)], [(2, 0), (2, 1)], [(3, 0), (3, 1)], [(3, 2), (4, 2)]]

  # splitting the room into two smaller rooms.
  # the robot can only access to the other room by going through a door in the middle or a corridor at the top
  walls = [[(width / 2, _), (width / 2 + 1, _)] for _ in range(1, height - 1) if _ != height / 2]
  
  # location, box1, box2, door1, door2, carpet, switch
  allLocations = [(x, y) for x in range(width) for y in range(height)]
  sSets = [allLocations] +\
          [[CLEAN, STEPPED] for _ in carpets] +\
          [[CLOSED, OPEN] for _ in doors] +\
          [[0, 1]] +\
          [range(horizon + 1)]
  
  # the robot can change its locations and manipulate the switch
  cIndices = cIndex + dIndex
  
  # the transition function depends on the following features
  dependentIdx = [lIndex] + dIndex + [sIndex, tIndex]

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
  
  def carpetOpGen(idx, carpet):
    def carpetOp(s, a):
      loc = s[lIndex]
      carpetState = s[idx]
      if loc == carpet: return STEPPED
      else: return carpetState
    return carpetOp
  
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
  
  # time elapses
  def timeOp(s, a):
    return s[-1] + 1

  tFunc = [navigate] +\
          [carpetOpGen(cIndexStart + i, carpets[i]) for i in range(cSize)] +\
          [doorOpGen(dIndexStart + i, doors[i]) for i in range(dSize)] +\
          [switchOp] +\
          [timeOp]

  s0List = [(0, 0)] +\
           [CLEAN for _ in carpets] +\
           [CLOSED for _ in doors] +\
           [ON, 0]
  s0 = tuple(s0List)
  
  terminal = lambda s: s[tIndex] == horizon
  gamma = .9

  # there is a reward of -1 at any step except when goal is reached
  # note that the domain of this function should not include any environmental features!
  bonus = util.Counter()
  for loc in allLocations: bonus[loc] = random.random() < .4

  def reward(s, a):
    if s[lIndex] == switch and s[sIndex] == ON and a == TURNOFFSWITCH:
      return 10
    else:
      # create some random rewards in the domain to break ties
      return bonus[s[lIndex]]
  rFunc = reward
  
  # the domain handler
  agent = ConsQueryAgent(sSets, aSets, rFunc, tFunc, s0, terminal, gamma, cIndices, dependentIdx)

  k = 3

  relFeats, domPis = agent.findRelevantFeaturesAndDomPis()

  start = time.time()
  agent.findMinimaxRegretPolicyQ(k, domPis)
  end = time.time()
  print end - start

  start = time.time()
  agent.findGlobalMinimaxRegretPolicyQ(k, domPis)
  end = time.time()
  print end - start

  #writeToFile('office.out', end - start)


def flatOfficNav():
  """
  An efficient and theoretically-unsound way to implement the office navigation domain.
  Rather than having features, we only have a set of possible constraints.
  """
  width = 10
  height = 10

  getRandLoc = lambda: (random.randint(0, width - 2), random.randint(0, height - 2))
  mdp = {}
  
  # some objects
  numOfCons = 20
  objectsInOneCons = 2
  
  mdp['s0'] = (0, 0)
  #switch = (8, 8)
  switch = getRandLoc()

  consSets = [[getRandLoc() for _ in range(objectsInOneCons)] for cons in range(numOfCons)]
  #consSets = [[(0, 1)], [(1, 1)], [(1, 0)]]

  print switch, consSets
  
  mdp['S'] = [(x, y) for x in range(width) for y in range(height)]
  #mdp['A'] = [(1, 0), (0, 1), (-1, 0), (0, -1)] 
  mdp['A'] = [(1, 0), (0, 1), (1, 1)] 
  
  # the posterior state of taking a in s without noise
  def move(s, a):
    moved = (s[0] + a[0], s[1] + a[1])
    if moved[0] >= 0 and moved[0] < width and moved[1] >= 0 and moved[1] < height:
      return moved
    else:
      return s

  """
  # stochastic transition
  def transit(s, a, sp):
    prob = 0
    for ap in mdp['A']:
      if ap == a:
        if sp == move(s, a): prob += 1 - noise
      elif move(s, ap) == sp:
        prob += noise / (len(mdp['A']) - 1)
    return prob
  """
  def transit(s, a, sp):
    return move(s, a) == sp

  mdp['T'] = transit
  
  def reward(s, a):
    if s == switch: return 0
    # don't bounce
    elif s == move(s, a): return -100
    else:
      for cons in consSets:
        # discourage the robot to occupy constraint-violating states unless necessary
        if s in cons: return -1.001
      return -1
      
  mdp['r'] = reward

  # terminates when the robot arrives at the switch or at the border
  mdp['terminal'] = lambda s: s == switch
  
  agent = ConsQueryAgent(mdp, consSets)

  start = time.time()
  if method == 'alg1':
    feats = agent.findRelevantFeaturesAndDomPis()
  elif method == 'alg3':
    feats = agent.findRelevantFeatsUsingHeu()
  elif method == 'brute':
    feats = agent.findRelevantFeatsBruteForce()
  else:
    raise Exception('unknown alg')
  end = time.time()
  elapsed = end - start

  print feats, elapsed
  
  writeToFile(method + 'Feats.out', len(feats))
  writeToFile(method + 'Time.out', elapsed)


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
