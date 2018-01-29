import easyDomains
from consQueryAgents import ConsQueryAgent
import time
import random
import numpy
import scipy
import pickle
import getopt
import sys
import os.path

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

def classicOfficNav(k, size, numOfCarpets, constrainHuman, dry, rnd):
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
  # specify the size of the domain, which are the robot's possible locations
  width = size
  height = size
  
  robot = (0, 0)
  switch = (width - 1, height - 1)
  
  horizon = width + height

  getBoundedRandLoc = lambda: (random.randint(1, width - 2), random.randint(1, height - 1))

  """
  # make sure carpets and robots are not covered by carpets
  def getBoundedRandLoc():
    while True:
      loc = (random.randint(0, width - 1), random.randint(0, height - 1))

      if loc == robot: continue
      else: return loc
  """

  #walls = [(width / 2, _) for _ in range(0, height) if _ != 0 and _ != height / 2]
  walls = []

  doors = []#[(width / 2, height / 2)]

  carpets = [getBoundedRandLoc() for _ in range(numOfCarpets)]

  lIndex = 0
  dIndexStart = lIndex + 1
  dSize = len(doors)
  sIndex = dIndexStart + dSize
  # time is needed when there are reversible features or a goal constraint
  tIndex = sIndex + 1
  
  dIndex = range(dIndexStart, dIndexStart + dSize)
  
  allLocations = [(x, y) for x in range(width) for y in range(height)]
  # cross product of possible values of all features
  # location, door1, door2, carpets, switch, time
  sSets = [allLocations] +\
          [[CLOSED, OPEN] for _ in doors] +\
          [[0, 1]] + [range(horizon)]
  
  directionalActs = [(1, 0), (0, 1), (1, 1)]
  aSets = directionalActs #+ [TURNOFFSWITCH]
 
  # check what the world is like
  for y in range(height):
    for x in range(width):
      if (x, y) in walls: print '[ X]',
      elif carpets.count((x, y)) == 1: print '[%2d]' % carpets.index((x, y)),
      elif carpets.count((x, y)) > 1: print '[%2d*' % carpets.index((x, y)),
      elif (x, y) == switch: print '[ S]',
      elif (x, y) == robot: print '[ R]',
      else: print '[  ]',
    print
  
  for _ in range(len(carpets)): print _, carpets[_]

  # factored transition function
  def navigate(s, a):
    loc = s[lIndex]
    if a in directionalActs:
      if loc[0] == width - 1 and loc[1] < height - 1: a = (0, 1)
      if loc[0] < width - 1 and loc[1] == height - 1: a = (1, 0)

      sp = (loc[0] + a[0], loc[1] + a[1])
      # not blocked by borders, closed doors or walls
      if (sp[0] >= 0 and sp[0] < width and sp[1] >= 0 and sp[1] < height) and\
         not any(s[idx] == CLOSED and sp == doors[idx - dIndexStart] for idx in dIndex) and\
         not sp in walls:
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
    #if loc == switch and a == 'turnOffSwitch': switchState = OFF 
    if loc == switch and a == (1, 1): switchState = OFF 
    return switchState
  
  # time elapses
  def timeOp(s, a):
    return s[tIndex] + 1

  tFunc = [navigate] +\
          [doorOpGen(dIndexStart + i, doors[i]) for i in range(dSize)] +\
          [switchOp] + [timeOp]

  s0List = [robot] +\
           [CLOSED for _ in doors] +\
           [ON, 0]
  s0 = tuple(s0List)
  
  terminal = lambda s: s[tIndex] == horizon
  #terminal = lambda s: s[lIndex] == switch

  def oldReward(s, a):
    if s[lIndex] == switch and s[sIndex] == ON and a == TURNOFFSWITCH:
      return 10
    else:
      # create some random rewards in the domain to break ties
      return 0

  carpetRewardDict = [-random.random() for _ in range(numOfCarpets)]
  def reward(s, a):
    if s[sIndex] == ON:
      if s[lIndex] in carpets:
        carpetId = carpets.index(s[lIndex])
        return carpetRewardDict[carpetId]
      else:
        return -1
    else:
      return 0
  
  def gradientReward(s, a):
    if s[sIndex] == ON:
      if s[lIndex] in carpets:
        return 0
      else:
        x, y = s[lIndex]
        return -(x + y)
    else:
      return 0
    
  locationRewardDict = {(x, y): -random.random() for x in range(width) for y in range(height)}
  def locationReward(s, a):
    if s[sIndex] == ON:
      if s[lIndex] in carpets:
        return 0
      else:
        return locationRewardDict[s[lIndex]]
    else:
      return 0
 
  rFunc = locationReward
  gamma = 1
  
  mdp = easyDomains.getFactoredMDP(sSets, aSets, rFunc, tFunc, s0, terminal, gamma)
  print mdp['S']

  # states that should not be visited
  # let's not make carpets features but constraints directly
  consStates = [[s for s in mdp['S'] if s[lIndex] == _] for _ in carpets]
  
  goalConsStates = filter(lambda s: s[sIndex] == ON and s[tIndex] >= horizon, mdp['S'])

  agent = ConsQueryAgent(mdp, consStates, goalConsStates, constrainHuman=constrainHuman)

  domainFileName = 'domain_' + str(numOfCarpets) + '_' + str(rnd) + '.pkl'
  if os.path.exists(domainFileName):
    data = pickle.load(open(domainFileName, 'rb'))
    if data == 'INITIALIZED':
      # failure in computing dom pi. do not try again.
      print "ABORT"
      return
    else:
      (relFeats, domPis, domPiTime) = data
  else:
    pickle.dump('INITIALIZED', open(domainFileName, 'wb'))

    # find dom pi (which may be used to find queries and will be used for evaluation)
    start = time.time()
    relFeats, domPis = agent.findRelevantFeaturesAndDomPis()
    end = time.time()
    domPiTime = end - start

    print "num of rel feats", len(relFeats)
    pickle.dump((relFeats, domPis, domPiTime), open(domainFileName, 'wb'))

  methods = ['alg1', 'chain', 'naiveChain', 'relevantRandom', 'random', 'nq']

  # decide the true changeable features for expected regrets
  numpy.random.seed(2 * (1 + rnd)) # avoid weird coupling, e.g., the ones that are queried are exactly the true changeable ones
  violableIndices = numpy.random.choice(range(len(agent.allCons)), k, replace=False)
  violableCons = [agent.allCons[_] for _ in violableIndices]

  for method in methods:
    start = time.time()
    if method == 'brute':
      q = agent.findMinimaxRegretConstraintQBruteForce(k, relFeats, domPis)
    elif method == 'reallyBrute':
      # really brute still need domPis to find out MR...
      q = agent.findMinimaxRegretConstraintQBruteForce(k, agent.allCons, domPis)
    elif method == 'alg1':
      q = agent.findMinimaxRegretConstraintQ(k, relFeats, domPis)
    elif method == 'alg1NoFilter':
      q = agent.findMinimaxRegretConstraintQ(k, relFeats, domPis, filterHeu=False)
    elif method == 'alg1NoScope':
      q = agent.findMinimaxRegretConstraintQ(k, relFeats, domPis, scopeHeu=False)
    elif method == 'naiveChain':
      q = agent.findChainedAdvConstraintQ(k, relFeats, domPis, informed=False)
    elif method == 'chain':
      q = agent.findChainedAdvConstraintQ(k, relFeats, domPis, informed=True)
    elif method == 'relevantRandom':
      q = agent.findRelevantRandomConstraintQ(k, relFeats)
    elif method == 'random':
      q = agent.findRandomConstraintQ(k)
    elif method == 'nq':
      q = []
    elif method == 'domPiBruteForce':
      # HACKING compute how long is needed to find a dominating policies by enumeration
      agent.findRelevantFeaturesBruteForce()
      q = []
    else:
      raise Exception('unknown method', method)
    end = time.time()

    runTime = end - start + (0 if method in ['random', 'nq'] else domPiTime)

    print method, q

    mrk, advPi = agent.findMRAdvPi(q, relFeats, domPis, k, consHuman=True)

    regret = agent.findRegret(q, violableCons)

    print mrk, regret, runTime
    
    if not dry:
      saveToFile(method, k, numOfCarpets, constrainHuman, q, mrk, runTime, regret)

def saveToFile(method, k, numOfCarpets, constrainHuman, q, mrk, runTime, regret):
  ret = {}
  ret['mrk'] = mrk
  ret['regret'] = regret
  ret['time'] = runTime
  ret['q'] = q

  postfix = 'mrk' if constrainHuman else 'mr'

  # not distinguishing mr and mrk in filenames, so use a subdirectory
  pickle.dump(ret, open(method + '_' + postfix + '_' + str(k) + '_' + str(numOfCarpets) + '_' + str(rnd) + '.pkl', 'wb'))

if __name__ == '__main__':
  # default values
  method = None
  k = 2
  constrainHuman = False
  dry = False # do not safe to files if dry run

  numOfCarpets = 10
  size = 6

  try:
    opts, args = getopt.getopt(sys.argv[1:], 's:k:n:cr:d')
  except getopt.GetoptError:
    raise Exception('Unknown flag')
  for opt, arg in opts:
    if opt == '-k':
      k = int(arg)
    elif opt == '-s':
      size = int(arg)
    elif opt == '-n':
      numOfCarpets = int(arg)
    elif opt == '-c':
      constrainHuman = True
    elif opt == '-d':
      dry = True
    elif opt == '-r':
      rnd = int(arg)

      random.seed(rnd)
      # not necessarily using the following packages, but just to be sure
      numpy.random.seed(rnd)
      scipy.random.seed(rnd)
      
      print 'random seed', rnd
    else:
      raise Exception('unknown argument')

  classicOfficNav(k, size, numOfCarpets, constrainHuman, dry, rnd)
