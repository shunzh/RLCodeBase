import easyDomains
from consQueryAgents import ConsQueryAgent
import time
import random
import numpy
import scipy
import pickle
import getopt
import sys
import itertools

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

def classicOfficNav(method, k, numOfCarpets, typeOfQuery='cons', constrainHuman=False, portionOfViolableCons=0):
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
  # leave the left column and the top row empty
  getBoundedRandLoc = lambda: (random.randint(2, width - 3), random.randint(0, height - 2))

  # specify the size of the domain, which are the robot's possible locations
  width = 7
  height = 7
  #horizon = width + height * 2
  
  doors = [(1, 0), (width - 2, 0)]

  switch = (width - 1, 0)

  carpets = [getBoundedRandLoc() for _ in range(numOfCarpets)]
  #carpets = []
  
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
  #walls = [(width / 2, _) for _ in range(height) if (width / 2, _) not in doors]
  #walls = []
  
  # half circle
  walls = [(1, _) for _ in range(1, height / 2)] +\
          [(width - 2, _) for _ in range(1, height / 2)] +\
          [(_, height / 2) for _ in range(1, width - 1)]
  
  # location, box1, box2, door1, door2, carpet, switch
  allLocations = [(x, y) for x in range(width) for y in range(height)]
  sSets = [allLocations] +\
          [[CLOSED, OPEN] for _ in doors] +\
          [[0, 1]]
  
  navASets = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)]
  aSets = navASets + [OPENDOOR, CLOSEDOOR, TURNOFFSWITCH]
  
  for y in range(height):
    for x in range(width):
      if (x, y) in walls: print 'X',
      elif (x, y) in carpets: print carpets.index((x, y)),
      elif (x, y) in doors: print numOfCarpets + doors.index((x, y)),
      elif (x, y) == switch: print 'S',
      else: print ' ',
    print
 
  # factored transition function
  def navigate(s, a):
    loc = s[lIndex]
    if a in navASets:
      sp = (loc[0] + a[0], loc[1] + a[1])
      # not blocked by borders, closed doors or walls
      if (sp[0] >= 0 and sp[0] < width and sp[1] >= 0 and sp[1] < height) and\
         not any(s[idx] == CLOSED and sp == doors[idx - dIndexStart] for idx in dIndex) and\
         sp not in walls:
        return sp
    return loc
  
  def doorOpGen(idx, door):
    def doorOp(s, a):
      loc = s[lIndex]
      doorState = s[idx]
      if a in [OPENDOOR, CLOSEDOOR]:
        # the robot needs to be close to a door to operate on it
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

  # initially, the door in the middle is closed
  # (which may be necessary for the robot to query to find a shorter path at least)
  # and the door at the top is open (so the robot can always find a feasible policy without querying)
  s0List = [(0, 0)] +\
           [CLOSED, OPEN] +\
           [ON]
  s0 = tuple(s0List)
  
  # terminate only when horizon reaches
  terminal = lambda s: s[sIndex] == OFF
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
  irreversibleSet = range(len(consStates))
  consStates += [[s for s in mdp['S'] if s[idx] != mdp['s0'][idx]] for idx in range(dIndexStart, dIndexStart + len(doors))]
  agent = ConsQueryAgent(mdp, consStates, irreversibleSet, constrainHuman)

  start = time.time()

  def findQuerySpace(cons, k, typeOfQuery):
    if typeOfQuery == 'cons':
      # return all possible subset of cons
      if len(cons) > k:
        return list(itertools.combinations(cons, k))
      else:
        # only one query to consider, that is asking about all constraints
        return [tuple(cons)]
    elif typeOfQuery == 'feats':
      feats = set([_[1] for _ in cons])
      if len(feats) > k:
        return [tuple(filter(lambda _: _[1] in feats, cons))
                for feats in itertools.combinations(feats, k)]
      else:
        # only one query to consider, that is asking about all features
        return [tuple(cons)]
    else:
      raise Exception('unknown type of query ' + str(typeOfQuery))
      
  if method == 'brute':
    relCons, domPis = agent.findRelevantConsAndDomPis()
    # way to slow to consider all constraints, so consider relevant constraints.
    queries = findQuerySpace(relCons, k, typeOfQuery)
    q = agent.findMinimaxRegretQ(queries, relCons, domPis, pruning=False)
  elif method == 'alg1':
    relCons, domPis = agent.findRelevantConsAndDomPis()
    queries = findQuerySpace(relCons, k, typeOfQuery)
    print 'queries', queries
    q = agent.findMinimaxRegretQ(queries, relCons, domPis, pruning=True)
  elif method == 'chain':
    relCons, domPis = agent.findRelevantConsAndDomPis()
    q = agent.findChaindAdvQ(k, relCons, domPis)
  elif method == 'random':
    queries = findQuerySpace(agent.allCons, k, typeOfQuery)
    q = random.choice(queries)
  elif method == 'nq':
    q = []
  else:
    raise Exception('unknown method', method)

  end = time.time()

  # we may need relFeats and domPis for evaluation. they are not timed.
  if 'relCons' not in vars() or 'domPis' not in vars():
    relCons, domPis = agent.findRelevantConsAndDomPis()

  mr, advPi = agent.findMRAdvPi(q, relCons, domPis)

  #indices = numpy.random.choice(range(len(relFeats)), len(relFeats) * violableRatio)
  #violableCons = [list(relFeats)[_] for _ in indices]
  
  print q, mr

  return mr, end - start
  """
  violableIndices = numpy.random.choice(range(len(agent.allCons)), int(len(agent.allCons) * portionOfViolableCons), replace=False)
  violableCons = [agent.allCons[_] for _ in violableIndices]
  
  print 'violable', violableCons
  
  regret = agent.findRegret(q, violableCons)
  print 'regret', regret
  return regret, end - start
  """


if __name__ == '__main__':
  method = 'alg1'
  k = 1
  numOfCarpets = 5
  #typeOfQuery = 'feats'
  typeOfQuery = 'cons'
  constrainHuman = False

  try:
    opts, args = getopt.getopt(sys.argv[1:], 'a:k:n:p:r:')
  except getopt.GetoptError:
    raise Exception('Unknown flag')
  for opt, arg in opts:
    if opt == '-a':
      method = arg
    elif opt == '-k':
      k = int(arg)
    elif opt == '-n':
      numOfCarpets = int(arg)
    elif opt == '-p':
      ratioOfViolable = float(arg)
    elif opt == '-r':
      rnd = int(arg)

      random.seed(rnd)
      numpy.random.seed(rnd)
      scipy.random.seed(rnd)
    else:
      raise Exception('unknown argument')

  ret = {}
  
  #ret = pickle.load(open(method + '_' + str(k) + '_' + str(numOfCarpets) + '.pkl', 'rb'))

  #flatOfficNav()
  mr, timeElapsed = classicOfficNav(method, k, numOfCarpets, typeOfQuery, constrainHuman)

  ret['mr'] = mr
  ret['time'] = timeElapsed

  if 'ratioOfViolable' in vars():
    pickle.dump(ret, open(method + '_' + str(k) + '_' + str(numOfCarpets) + '_' + str(ratioOfViolable) + '_' + str(rnd) + '.pkl', 'wb'))
  else:
    pickle.dump(ret, open(method + '_' + str(k) + '_' + str(numOfCarpets) + '_' + str(rnd) + '.pkl', 'wb'))
