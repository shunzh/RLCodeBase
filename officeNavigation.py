import easyDomains
from consQueryAgents import ConsQueryAgent
import time
import getopt
import sys
import random
import numpy
import scipy

OPEN = 1
CLOSED = 0

STEPPED = 1
CLEAN = 0

ON = 1
OFF = 0 

def classicOfficNav():
  """
  The office navigation domain specified in the report using a factored representation.
  There are state factors indicating whether some carpets are dirty.
  However, this domain is hard to scale up since we have control variables |S|x|A|.
     _________
  2 |C|     |S|
  1 |  __C__  |
  0 |R___C____|
     0 1 2 3 4
  """
  # specify the size of the domain, which are the robot's possible locations
  width = 5
  height = 3
  
    # some objects
  boxes = [(0, 2), (1, 2), (2, 2), (3, 2)]
  #door1 = (1, 1)
  #door2 = (3, 1)
  switch = (width - 1, height - 1)

  LOCATION = 0
  SWITCH = len(boxes) + 1
  
  # pairs of adjacent locations that are blocked by a wall
  #walls = [[(0, 2), (1, 2)], [(1, 0), (1, 1)], [(2, 0), (2, 1)], [(3, 0), (3, 1)], [(3, 2), (4, 2)]]
  walls = []
  
  # location, box1, box2, door1, door2, carpet, switch
  sSets = [[(x, y) for x in range(width) for y in range(height)]] +\
          [[0, 1] for _ in boxes] +\
          [[0, 1]] #switch
  
  # the robot can change its locations and manipulate the switch
  cIndices = range(1, SWITCH) # location is not a constraint

  aSets = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1),
           #'openDoor', 'closeDoor',
           'turnOffSwitch']
 
  def move(s, a):
    loc = s[LOCATION]
    if type(a) == tuple:
      sp = (loc[0] + a[0], loc[1] + a[1])
      if sp[0] >= 0 and sp[0] < width and sp[1] >= 0 and sp[1] < height:
        # so it's not out of the border
        if True: #not (s[DOOR1] == CLOSED and sp == door1 or s[DOOR2] == CLOSED and sp == door2):
          # doors are fine
          blockedByWall = any(loc in wall and sp in wall for wall in walls)
          if not blockedByWall: return sp
    return loc
  
  def stepOnBoxGen(idx, box):
    def stepOnBox(s, a):
      loc = s[LOCATION]
      boxState = s[idx]
      if loc == box: return STEPPED
      else: return boxState
    return stepOnBox
  
  def doorOpGen(idx, door):
    def doorOp(s, a):
      loc = s[LOCATION]
      doorState = s[idx]
      if loc in [(door[0] - 1, door[1]), (door[0], door[1])]:
        if a == 'closeDoor': doorState = CLOSED
        elif a == 'openDoor': doorState = OPEN
        # otherwise the door state is unchanged
      return doorState
    return doorOp
  
  def switchOp(s, a):
    loc = s[LOCATION]
    switchState = s[SWITCH]
    if loc == switch and a == 'turnOffSwitch': switchState = OFF 
    return switchState

  tFunc = [move] +\
          [stepOnBoxGen(i+1, boxes[i]) for i in range(len(boxes))] +\
          [switchOp]

  s0List = [(0, 0)] +\
           [CLEAN for _ in range(len(boxes))] +\
           [ON] # switch is on
  s0 = tuple(s0List)
  
  terminal = lambda s: s[SWITCH] == OFF

  # there is a reward of -1 at any step except when goal is reached
  # note that the domain of this function should not include any environmental features!
  rFunc = lambda s, a: 0 if s[SWITCH] == OFF else -1
  
  # the domain handler
  officeNav = easyDomains.getFactoredMDP(sSets, aSets, rFunc, tFunc, s0, terminal)
  
  agent = ConsQueryAgent(officeNav, cIndices)

  start = time.time()
  agent.findRelevantFeatures()
  end = time.time()
  writeToFile('milp.out', end - start)

  """
  start = time.time()
  agent.findDominatingPoliciesBruteForce()
  end = time.time()
  writeToFile('brute.out', end - start)
  """


def flatOfficNav():
  width = 10
  height = 5

  getRandLoc = lambda: (random.randint(0, width - 1), random.randint(0, height - 1))
  mdp = {}
  
  # some objects
  numOfCons = 5
  objectsInOneCons = 5

  #mdp['s0'] = (4, 4)
  mdp['s0'] = getRandLoc()
  #switch = (4, 6)
  switch = getRandLoc()

  mdp['S'] = [(x, y) for x in range(width) for y in range(height)]
  mdp['A'] = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)] 
  
  def move(s, a, sp):
    moved = (s[0] + a[0], s[1] + a[1])
    if moved[0] >= 0 and moved[0] < width and moved[1] >= 0 and moved[1] < height:
      return sp == moved
    return s == sp # bounced back
  mdp['T'] = move
  mdp['r'] = lambda s, a: -1 if s != switch else 0

  mdp['terminal'] = lambda s: s == switch # terminates when the robot arrives at the switch
  
  consSets = [[getRandLoc() for _ in range(objectsInOneCons)] for cons in range(numOfCons)]
  #consSets = [[(4, 5)], [(0, 0)]]

  agent = ConsQueryAgent(mdp, consSets)

  start = time.time()
  print agent.findRelevantFeatures()
  #print agent.findRelevantFeatsBruteForce()
  end = time.time()
  print end - start


def writeToFile(name, value):
  f = open(name, 'w') # not appending
  f.write(str(value))
  f.close()


if __name__ == '__main__':
  try:
    opts, args = getopt.getopt(sys.argv[1:], 'r:')
  except getopt.GetoptError:
    raise Exception('Unknown flag')
  for opt, arg in opts:
    if opt == '-r':
      random.seed(int(arg))
      numpy.random.seed(int(arg))
      scipy.random.seed(int(arg))
 
  flatOfficNav()
