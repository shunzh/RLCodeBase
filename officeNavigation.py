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

class Spec():
  """
  A object that get a dictionary and convert them into attributes
  """
  def __init__(self, adict):
    self.__dict__.update(adict)

def sampleWrold():
  """
  2 | |     |S|
  1 |  D_C_D  |
  0 |R___C____|
     0 1 2 3 4
  """
  pass

def squareWorld(size, numOfCarpets):
  """
  Squared world with width = height = size.
  The robot and the swtich are at opposite corners.
  No walls or doors.
  """
  width = size
  height = size
  
  robot = (0, 0)
  switch = (width - 1, height - 1)
  
  walls = []
  doors = []

  getBoundedRandLoc = lambda: (random.randint(0, width - 2), random.randint(1, height - 1))
  carpets = [getBoundedRandLoc() for _ in range(numOfCarpets)]
 
  # create a Spec object so that the varibales here are properties
  return Spec({(var, eval(var)) for var in ['width', 'height', 'robot', 'switch', 'walls', 'doors', 'carpets']})

def classicOfficNav(spec, k, constrainHuman, dry, rnd):
  """
  The office navigation domain specified in the report using a factored representation.
  There are state factors indicating whether some carpets are dirty.
    
  """
  # don't want to use locals.update.. otherwise would be hard to debug
  horizon = spec.width + spec.height

  lIndex = 0
  dIndexStart = lIndex + 1
  dSize = len(spec['doors'])
  sIndex = dIndexStart + dSize
  # time is needed when there are reversible features or a goal constraint
  tIndex = sIndex + 1
  
  dIndex = range(dIndexStart, dIndexStart + dSize)
  
  allLocations = [(x, y) for x in range(spec.width) for y in range(spec.height)]
  # cross product of possible values of all features
  # location, door1, door2, carpets, switch, time
  sSets = [allLocations] +\
          [[CLOSED, OPEN] for _ in spec.doors] +\
          [[0, 1]]
  
  directionalActs = [(1, 0), (0, 1), (1, 1)]
  aSets = directionalActs + [TURNOFFSWITCH]
 
  # check what the world is like
  for y in range(spec.height):
    for x in range(spec.width):
      if (x, y) in spec.walls: print '[ X]',
      elif spec.carpets.count((x, y)) == 1: print '[%2d]' % spec.carpets.index((x, y)),
      elif spec.carpets.count((x, y)) > 1: print '[%2d*' % spec.carpets.index((x, y)),
      elif (x, y) == spec.switch: print '[ S]',
      elif (x, y) == spec.robot: print '[ R]',
      else: print '[  ]',
    print
  
  for _ in range(len(spec.carpets)): print _, spec.carpets[_]

  # factored transition function
  def navigate(s, a):
    loc = s[lIndex]
    if a in directionalActs:
      sp = (loc[0] + a[0], loc[1] + a[1])
      # not blocked by borders, closed doors or walls
      if (sp[0] >= 0 and sp[0] < spec.width and sp[1] >= 0 and sp[1] < spec.height) and\
         not any(s[idx] == CLOSED and sp == spec.doors[idx - dIndexStart] for idx in dIndex) and\
         not sp in spec.walls:
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
    if loc == spec.switch and a == 'turnOffSwitch': switchState = OFF 
    return switchState
  
  tFunc = [navigate] +\
          [doorOpGen(dIndexStart + i, spec.doors[i]) for i in range(dSize)] +\
          [switchOp]

  s0List = [spec.robot] +\
           [CLOSED for _ in spec.doors] +\
           [ON]
  s0 = tuple(s0List)
  
  terminal = lambda s: s[lIndex] == spec.switch

  def oldReward(s, a):
    if s[lIndex] == spec.switch and s[sIndex] == ON and a == TURNOFFSWITCH:
      return 1
    else:
      # create some random rewards in the domain to break ties
      return 0
 
  # a list of possible reward functions
  carpetRewardDict = [-random.random() for _ in range(numOfCarpets)]
  def reward(s, a):
    if s[sIndex] == ON:
      if s[lIndex] in spec.carpets:
        carpetId = spec.carpets.index(s[lIndex])
        return carpetRewardDict[carpetId]
      else:
        return -1
    else:
      return 0
  
  def gradientReward(s, a):
    if s[sIndex] == ON:
      if s[lIndex] in spec.carpets:
        return 0
      else:
        x, y = s[lIndex]
        return -(x + y)
    else:
      return 0
    
  # the reward is 0 when the robot is on a carpet, and a pre-specified random reward otherwise.
  locationRewardDict = {(x, y): -random.random() for x in range(spec.width) for y in range(spec.height)}
  def locationReward(s, a):
    if s[sIndex] == ON:
      if s[lIndex] in spec.carpets:
        return 0
      else:
        return locationRewardDict[s[lIndex]]
    else:
      return 0
 
  rFunc = locationReward
  gamma = 1

  mdp = easyDomains.getFactoredMDP(sSets, aSets, rFunc, tFunc, s0, terminal, gamma)

  # states that should not be visited
  # let's not make carpets features but constraints directly
  consStates = [[s for s in mdp['S'] if s[lIndex] == _] for _ in spec.carpets]
  
  #goalConsStates = filter(lambda s: s[sIndex] == ON and s[tIndex] >= horizon, mdp['S'])

  agent = ConsQueryAgent(mdp, consStates, constrainHuman=constrainHuman)

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
