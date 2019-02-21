import easyDomains
from consQueryAgents import ConsQueryAgent, GreedyForSafetyAgent,\
  DomPiHeuForSafetyAgent, MaxProbSafePolicyExistAgent, RandomQueryForSafetyAgent,\
  OptQueryForSafetyAgent
import time
import random
import numpy
import scipy
import pickle
import getopt
import sys
import os.path

# some consts
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

TERMINATE = 'terminate'

class Spec():
  """
  A object that describes the specifications of an MDP
  """
  def __init__(self, width, height, robot, switch, walls, doors, boxes, carpets, horizon):
    self.width = width
    self.height = height
    
    self.robot = robot
    self.switch = switch
    
    self.walls = walls
    self.doors = doors
    self.boxes = boxes
    self.carpets = carpets
    
    self.horizon = horizon

def toyWrold():
  # just make plotting easier
  R = 'R'
  _ = '_'
  W = 'W'
  C = 'C'
  S = 'S'
  D = 'D'
  B = 'B'

  # layout of the domain
  map = [[R, C, _, C, _],
         [_, W, _, W, _],
         [_, W, _, C, S]]

  width = len(map[0])
  height = len(map)
  
  walls = []
  doors = []
  carpets = []
  boxes = []

  for i in range(height):
    for j in range(width):
      if map[i][j] == R:
        robot = (j, i)
      elif map[i][j] == S:
        switch = (j, i)
      elif map[i][j] == W:
        walls.append((j, i))
      elif map[i][j] == D:
        doors.append((j, i))
      elif map[i][j] == C:
        carpets.append((j, i))
      elif map[i][j] == B:
        boxes.append((j, i))
 
  horizon = width + height
 
  return Spec(width, height, robot, switch, walls, doors, boxes, carpets, horizon);

def squareWorld(size, numOfCarpets, avoidBorder=True):
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

  if avoidBorder:
    admissibleLocs = [(x, y) for x in range(width - 1) for y in range(1, height)]
  else:
    admissibleLocs = [(x, y) for x in range(width) for y in range(height)]

  assert len(admissibleLocs) >= numOfCarpets
  carpetIndices = numpy.random.choice(range(len(admissibleLocs)), numOfCarpets, replace=False)
  carpets = [admissibleLocs[carpetIndex] for carpetIndex in carpetIndices]

  boxes = []
  
  horizon = width + height
 
  return Spec(width, height, robot, switch, walls, doors, boxes, carpets, horizon);

def toySokobanWorld():
  """
  A very small domain that can be solved quickly. For sanity check.
     _________
  1 |  X      |
  0 |R_B_____S|
     0 1 2 3 4
  """
  width = 5
  height = 2
  
  robot = (0, 0)
  switch = (width - 1, 0)
  
  walls = [(1, 1)]
  doors = []
  
  boxes = [(1, 0)]
  carpets = [] # no non-reversible features
  
  horizon = 10
  
  return Spec(width, height, robot, switch, walls, doors, boxes, carpets, horizon);

def sokobanWorld():
  """
  A domain as a motivating example in the reports.
     ___________________
  2 |  X       X X      |
  1 |  X       X X      |
  0 |R_B_______B_______S|
     0 1 2 3 4 5 6 7 8 9
  """
  width = 10
  height = 3
  
  robot = (0, 0)
  switch = (width - 1, 0)
  
  walls = [(x, y) for x in [1, 5, 6] for y in [1, 2]]
  doors = []
  
  boxes = [(1, 0), (5, 0)]
  carpets = [] # no non-reversible features
  
  horizon = 20
  
  return Spec(width, height, robot, switch, walls, doors, boxes, carpets, horizon);

def parameterizedSokobanWorld(size, numOfBoxes):
  """
  """
  width = height = size
  
  robot = (0, 0)
  switch = (width - 1, height - 1)
  
  walls = []
  doors = []
  
  boxes = [random.choice([(x, y) for x in range(width) for y in range(height) if x != 0 or y != 0]) for _ in range(numOfBoxes)]
  carpets = [] # no non-reversible features
  
  horizon = size * 2
  
  return Spec(width, height, robot, switch, walls, doors, boxes, carpets, horizon);

def classicOfficNav(spec, k, constrainHuman, dry, rnd, consProbs=None):
  """
  spec: specification of the factored mdp
  k: number of queries (in batch querying setting
  constrainHuman: a flag controls MR vs MR_k
  dry: no output to file if True
  rnd: random seed
  consProbs: only for Bayesian setting. ["prob that ith unknown feature is free" for i in range(self.numOfCons)]
    If None (by default), set randomly
  """
  # need to flatten the state representation to a vector.
  # (robot's location, doors, boxes, switch, time)
  
  # robot's location
  lIndex = 0
  
  # door indices
  dIndexStart = lIndex + 1
  dSize = len(spec.doors)

  # box indices
  bIndexStart = dIndexStart + dSize
  bSize = len(spec.boxes)

  # switch index
  sIndex = bIndexStart + bSize

  # time index
  # time is needed when there are horizon-dependent constraints
  tIndex = sIndex + 1
  
  dIndices = range(dIndexStart, dIndexStart + dSize)
  bIndices = range(bIndexStart, bIndexStart + bSize)
  
  #directionalActs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
  directionalActs = [(1, 0), (0, 1)]
  aSets = directionalActs + [TURNOFFSWITCH]
 
  # check what the world is like
  for y in range(spec.height):
    for x in range(spec.width):
      if (x, y) in spec.walls: print '[ X]',
      elif spec.carpets.count((x, y)) == 1: print '[%2d]' % spec.carpets.index((x, y)),
      elif spec.carpets.count((x, y)) > 1: print '[%2d*' % spec.carpets.index((x, y)),
      elif (x, y) in spec.boxes: print '[ B]',
      elif (x, y) == spec.switch: print '[ S]',
      elif (x, y) == spec.robot: print '[ R]',
      else: print '[  ]',
    print
  
  print 'carpets'
  for _ in range(len(spec.carpets)): print _, spec.carpets[_]

  def boxMovable(idx, s, a):
    """
    return True if the box represented by s[idx] can be moved with a applied in state s
    False otherwise
    """
    assert a in directionalActs

    box = s[idx]
    boxP = (box[0] + a[0], box[1] + a[1]) # box prime, the next location without considering constraints
    # box is not moved across the border and not into walls or other boxes
    if boxP[0] >= 0 and boxP[0] < spec.width and boxP[1] >= 0 and boxP[1] < spec.height\
       and not boxP in spec.walls\
       and not boxP in spec.boxes:
      return True
    else:
      return False
 
  # factored transition function
  def navigate(s, a):
    loc = s[lIndex]
    if a in directionalActs:
      sp = (loc[0] + a[0], loc[1] + a[1])
      # not blocked by borders, closed doors
      # not pushing towards a non-movable box
      # not blocked by walls
      if (sp[0] >= 0 and sp[0] < spec.width and sp[1] >= 0 and sp[1] < spec.height) and\
         not any(s[idx] == CLOSED and sp == spec.doors[idx - dIndexStart] for idx in dIndices) and\
         not any(sp == s[idx] and not boxMovable(idx, s, a) for idx in bIndices) and\
         not sp in spec.walls:
        return sp
    # otherwise the agent is blocked and return the unchanged location
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
 
  def boxOpGen(idx):
    def boxOp(s, a):
      loc = s[lIndex]
      box = s[idx]
      if a in directionalActs and navigate(s, a) == box:
        if boxMovable(idx, s, a):
          newBox = (box[0] + a[0], box[1] + a[1])
          return newBox
      # otherwise the box state is unchanged
      return box
    return boxOp

  def switchOp(s, a):
    loc = s[lIndex]
    switchState = s[sIndex]
    if loc == spec.switch and a == 'turnOffSwitch': switchState = OFF 
    return switchState
  
  def timeElapse(s, a):
    return s[tIndex] + 1
  
  
  # all physically possible locations
  allLocations = [(x, y) for x in range(spec.width) for y in range(spec.height) if (x, y) not in spec.walls]
  # cross product of possible values of all features
  # location, door1, door2, carpets, switch, time
  sSets = [allLocations] +\
          [[CLOSED, OPEN] for _ in spec.doors] +\
          [allLocations for _ in spec.boxes] +\
          [[OFF, ON]]
 
  # the transition function is also factored, each item is a function defined on S x A -> S_i
  tFunc = [navigate] +\
          [doorOpGen(i, spec.doors[i - dIndexStart]) for i in dIndices] +\
          [boxOpGen(i) for i in bIndices] +\
          [switchOp]

  s0List = [spec.robot] +\
           [CLOSED for _ in spec.doors] +\
           spec.boxes +\
           [ON]
  s0 = tuple(s0List)
  
  terminal = lambda s: s[lIndex] == spec.switch
  #terminal = lambda s: s[tIndex] == spec.horizon

  # a list of possible reward functions
  # using locationReward in the IJCAI paper, where the difference between our algorithm and baselines are maximized
  # because there are different costs of locations where carpets are not covered, so it is crucial to decide which states should avoid blah blah
  def oldReward(s, a):
    if s[lIndex] == spec.switch and s[sIndex] == ON and a == TURNOFFSWITCH:
      return 1
    else:
      # create some random rewards in the domain to break ties
      return 0
 
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
  
  # reward based on whether the constraints (goalCons) are satisfied
  # reward = 1 if the robot takes a termination action and the current state satisfies the constraints.
  def goalConstrainedReward(goalCons):
    def reward(s, a):
      if a == TERMINATE and goalCons(s):
        return 1
      else:
        return 0
    
    return reward
 
  rFunc = oldReward
  # only give reward of 1 if the switch is turned off and the boxes are in their initial locations
  #rFunc = goalConstrainedReward(lambda s: s[sIndex] == OFF and all(s[bIdx] == s0[bIdx] for bIdx in bIndices))
  gamma = 1

  mdp = easyDomains.getFactoredMDP(sSets, aSets, rFunc, tFunc, s0, gamma, terminal)

  """
  consStates is [[states that violate the i-th constraint] for i in all constraints]
  Note that implementation here does not distinguish free and need-to-be-reverted features
  since we implement them both as constraints in linear programming anyway.

  Free features:
    require changing such features (e.g. making a carpet from clean to dirty) to be forbidden to visit
  Need-to-be-reverted features:
    first require time horizon to be added to the state representation
    then states where the features are not inverted are forbidden.
  """
  # carpet constraints:
  carpetCons = [[s for s in mdp.S if s[lIndex] == _] for _ in spec.carpets]
  
  # box constraints: by default, regard them as need-to-be-reverted features
  boxCons = [[s for s in mdp.S if terminal(s) and s[bIdx] != s0[bIdx]] for bIdx in bIndices]
  
  consStates = carpetCons + boxCons
  numOfCons = len(consStates)
  
  if consProbs == None:
    consProbs = [random.random() for _ in range(numOfCons)]
  print 'consProbs', zip(range(numOfCons), consProbs)

  agent = ConsQueryAgent(mdp, consStates, consProbs=consProbs, constrainHuman=constrainHuman)

  # true free features, randomly generated
  trueFreeFeatures = filter(lambda idx: random.random() < consProbs[idx], range(numOfCons))
  # if require existence of safe policies after querying: setting relevant features of a dominating policy to be free features
  #relFeats, domPis = agent.findRelevantFeaturesAndDomPis()
  #trueFreeFeatures = agent.findViolatedConstraints(random.choice(domPis))
  # or hand designed
  print 'true free features', trueFreeFeatures

  if not agent.initialSafePolicyExists():
    print 'initial policy does not exist'
    
    methods = ['iisAndRelpi', 'iisOnly', 'relpiOnly', 'maxProb', 'piHeu', 'random']
    #methods = ['opt']

    for method in methods:
      print method
      
      if method == 'iisAndRelpi':
        agent = GreedyForSafetyAgent(mdp, consStates, consProbs=consProbs, useIIS=True, useRelPi=True)
      elif method == 'iisOnly':
        agent = GreedyForSafetyAgent(mdp, consStates, consProbs=consProbs, useIIS=True, useRelPi=False)
      elif method == 'relpiOnly':
        agent = GreedyForSafetyAgent(mdp, consStates, consProbs=consProbs, useIIS=False, useRelPi=True)
      elif method == 'maxProb':
        agent = MaxProbSafePolicyExistAgent(mdp, consStates, consProbs=consProbs)
      elif method == 'piHeu':
        agent = DomPiHeuForSafetyAgent(mdp, consStates, consProbs=consProbs)
      elif method == 'random':
        agent = RandomQueryForSafetyAgent(mdp, consStates, consProbs=consProbs)
      elif method == 'opt':
        agent = OptQueryForSafetyAgent(mdp, consStates, consProbs=consProbs)
      else:
        raise Exception('unknown method', method)

      # keep the features the robot queried about for evaluation
      queries = []

      # it should not query more than the number of total features anyway..
      # but in case of bugs, this should not be a dead loop
      while len(queries) < len(consStates) + 1:
        query = agent.findQuery()

        if query == None:
          # the agent stops querying
          break
        elif query in trueFreeFeatures:
          agent.updateFeats(newFreeCon=query)
        else:
          agent.updateFeats(newLockedCon=query)
          
        queries.append(query)
        
      print 'queries', queries

      if not dry:
        # write to file
        f = open(method + '.out',"a")
        f.write(str(len(queries)) + '\n')
        f.close()
  else:
    print 'initial policy exists'
    # we bookkeep the dominating policies for all domains. check whether if we have already computed them.
    # if so we do not need to compute them again.
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
      # don't save anything if we are dryrun
      if not dry:
        pickle.dump('INITIALIZED', open(domainFileName, 'wb'))

      # find dom pi (which may be used to find queries and will be used for evaluation)
      start = time.time()
      relFeats, domPis = agent.findRelevantFeaturesAndDomPis()
      end = time.time()
      domPiTime = end - start

      print "num of rel feats", len(relFeats)
      
      if not dry:
        pickle.dump((relFeats, domPis, domPiTime), open(domainFileName, 'wb'))

    methods = ['alg1', 'chain', 'naiveChain', 'relevantRandom', 'random', 'nq']

    # decide the true changeable features for expected regrets
    numpy.random.seed(2 * (1 + rnd)) # avoid weird coupling, e.g., the ones that are queried are exactly the true changeable ones
    if len(agent.allCons) < k:
      raise Exception('k is larger than the number of unknown features so no need to select queries. abort.')
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
  k = 1
  constrainHuman = False
  dry = False # do not safe to files if dry run

  numOfCarpets = 10
  numOfBoxes = 0
  size = 5

  rnd = 0 # set a dummy random seed if no -r argument

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
      # disable dry run if output to file
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

  # test a hand-designed domain
  #classicOfficNav(toyWrold(), k, constrainHuman, dry, rnd, consProbs=[.9, .9, .9])

  # avoid border to make sure safe policies exist
  #consProbs = [.1,] * numOfCarpets
  classicOfficNav(squareWorld(size, numOfCarpets, avoidBorder=False), k, constrainHuman, dry, rnd)
  
  # good for testing irreversible features
  #classicOfficNav(sokobanWorld(), k, constrainHuman, dry, rnd)
  #classicOfficNav(toySokobanWorld(), k, constrainHuman, dry, rnd)
  #classicOfficNav(parameterizedSokobanWorld(size, numOfBoxes), k, constrainHuman, dry, rnd)
