import random
import sys
import mdp
import environment
import mdpEnvironment
import util
import optparse
import featureExtractors

import numpy as np
import numpy.linalg
import warnings

from graphics import *

from game import Actions

class ContinuousWorld(mdp.MarkovDecisionProcess):
  """
  A MDP that captures continuous state space, while the agent moves in discrete steps.

  State: (location, orientation) # orientation is dummy here
  Action: 8 directional movement, with a fixed step size.
  Transition: trivial.
  Reward: upon reaching a target / obstacle, obtain the corresponding reward.
          upon reaching a segment, if this segment is newer than the one visited before (encoded in the state space),
          then obtain the reward.
  """
  def __init__(self, init):
    """
    Args:
      init: a dict, that has attributes to be appended to self
            objs, boundary, radius.
    """
    # add necessary domains
    self.__dict__.update(init)

    # reward values that getReward will use
    self.rewards = {'targs': 1, 'obsts': -1, 'segs': 0.1, 'elevators': 0}

    # parameters
    self.livingReward = 0.0
    self.noise = 0.0 # DUMMY - need assumption on what it means to be noisy

  def getReachedObjects(self, l):
    """
    Determine whether a state is close to any object, within radius.
    Args:
      l: the loc to be checked.
      cond: constraint added
    Return:
      [(String, int)]: A list of the type of object that it's in the radius of, and its id.
                       Return [] if nothing matches.

    #FIXME not checked whether it's close to multiple objects. However, this won't happen in a valid domain.
    """
    ret = []

    for key, locs in self.objs.items():
      for idx in xrange(len(locs)):
        dist = numpy.linalg.norm(np.subtract(l, locs[idx]))
        if dist < self.radius:
          ret.append((key, idx))

    # if it's close to nothing
    return ret

  def getClosestTarget(self, l):
    """
    Get the closest target.
    """
    [minObj, minDist] = featureExtractors.getClosestObj(l, self.objs['targs'])
    return minObj

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

    This discretizes possible aciton set.
    """
    return ('north','west','south','east', 'ne', 'se', 'nw', 'sw')
    
  def getStates(self):
    """
    Return list of discrete random states. This is usually for sanity check.
    """
    states = []
    width = self.xBoundary[1] - self.xBoundary[0]
    height = self.xBoundary[1] - self.xBoundary[0]
    
    while len(states) < 400:
      x = self.xBoundary[0] + random.random() * width
      y = self.yBoundary[0] + random.random() * height
      states.append(((x, y), 0))
      
    return states
        
  def getReward(self, state, action, nextState):
    """
    Get reward for state, action, nextState transition.
    
    Note that the reward depends only on the state being
    departed (as in the R+N book examples, which more or
    less use this convention).
    """
    loc, orient = state
    nextLoc, newOrient = nextState
    
    reward = 0

    # check whether reaching a target or obstacle
    objInfoList = self.getReachedObjects(nextLoc)

    # any reached object applies
    for nextStateType, nextObjId in objInfoList:
      reward += self.rewards[nextStateType]

    return reward
      
  def clearObj(self, objType, objId):
    """
    Clear an object from self.objs, usually because the agent has got it.
    """
    del self.objs[objType][objId]

  def getStartState(self):
    """
    Start at the starting location, with no segment previously visited.
    """
    loc = self.objs['elevators'][0]
    return (loc, 0)
    
  def isFinal(self, state):
    """
    Check whether we should terminate at this state.

    Condition: reached exit elevator or no target left.
    """
    loc, orient = state

    # have trouble if two elevators are the same
    #objInfoList = self.getReachedObjects(loc)
    #return ('elevators', 1) in objInfoLists
    return len(self.objs['targs']) == 0 or len(self.objs['segs']) == 0

  def getTransitionStatesAndProbs(self, state, action):
    """
    Basically following physics laws, but considering:
    - bound back within self.xBoundary and self.yBoundary
    - change seg in the state representation upon reaching a new segment
    """
    loc, orient = state

    # move to new loc and check whether it's allowed
    newLoc = np.add(loc, np.multiply(self.step, Actions._directions[action]))
    if not self.isAllowed(newLoc):
      newLoc = loc

    newOrient = orient # doesn't change orient in this mdp
    
    successors = [((newLoc, newOrient), 1)]

    return successors                                
  
  def __aggregate(self, statesAndProbs):
    """
    Make sure stateAndProbs.keys() is a set (without duplicates)
    """
    counter = util.Counter()
    for state, prob in statesAndProbs:
      counter[state] += prob
    newStatesAndProbs = []
    for state, prob in counter.items():
      newStatesAndProbs.append((state, prob))
    return newStatesAndProbs
        
  def isAllowed(self, loc):
    """
    Check whether this state is valid
    """
    x, y = loc
    if x < self.xBoundary[0] or x >= self.xBoundary[1]: return False
    if y < self.yBoundary[0] or y >= self.yBoundary[1]: return False
    return True

def simpleToyDomain(category = 'targs'):
  """
  This domain can be configed using the category argument,
  so that we only have one target / obstacle for training.

  We put irrelevant objects in infPos that beyond the border,
  so that the agent cannot reach.
  """
  ret = {}

  size = 0.5
  # place that can't be reached
  infPos = (size + 1, size + 1)

  if category == 'targs':
    targs = [(size / 2, size / 2)]; obsts = [infPos]
  elif category == 'obsts':
    obsts = [(size / 2, size / 2)]; targs = [infPos]
  else:
    raise Exception("Undefined category.")

  segs = [infPos]

  # set the starting point to be random for training
  elevators = [(random.random() * size, random.random() * size), infPos] 

  ret['objs'] = {'targs': targs, 'obsts': obsts, 'segs': segs, 'elevators': elevators}

  ret['xBoundary'] = [0, size]
  ret['yBoundary'] = [0, size]

  ret['radius'] = 0.02
  ret['step'] = 0.004

  return ret

def toyDomain(category = 'targs'):
  """
  Similar to the simple toy domain, but with multiple (same) objects.
  """
  ret = {}

  layout = [(0.1 + 0.3 * x, 0.1 + 0.3 * y) for x in xrange(0, 3) for y in xrange(0, 3) ]
  infPos = (2, 2)

  if category == 'targs':
    targs = layout; obsts = [infPos]
  elif category == 'obsts':
    obsts = layout; targs = [infPos]
  segs = [infPos]

  elevators = [(0, 0), (1, 1)]
  ret['objs'] = {'targs': targs, 'obsts': obsts, 'segs': segs, 'elevators': elevators}

  ret['xBoundary'] = [-0.1, 1.1]
  ret['yBoundary'] = [-0.1, 1.1]

  # radius of an object (so the object doesn't appear as a point)
  ret['radius'] = 0.04

  # step size of the agent movement
  ret['step'] = 0.008

  return ret


def loadFromMat(filename, domainId):
  """
  Load from mat file that provided by Matt.

  Args:
    filename: name of the mat file, presumebaly in the same directory.
    domainId: there should be multiple configurations of rooms in this file,
              indicate which room to use.
  Return:
    no return, but add an obj attribute to self.
  """
  # read layout from source file
  s = util.loadmat(filename)

  ret = {}

  numObj = len(s['newRes']['all_objs']['id'])

  targs = []
  obsts = []
  segs = []
  elevators = []

  for idx in xrange(numObj):
    name = s['newRes']['all_objs']['id'][idx]

    x = s['newRes']['all_objs']['object_location']['x'][domainId][idx]
    y = s['newRes']['all_objs']['object_location']['z'][domainId][idx]

    if 'targ' in name:
      targs.append((x, y))
    elif 'obst' in name:
      obsts.append((x, y))
    elif 'seg' in name:
      # FIXME just guesses
      # drop far located segments
      if numpy.linalg.norm((x, y)) < 10:
        segs.append((x, y))
    elif 'elevator' in name:
      elevators.append((x, y))
    else:
      warnings.warn("Dropped unkown object typed '" + name + "' indexed at " + str(idx))

  if len(elevators) < 2:
    raise Exception("Elevators cannot be undefined.")

  ret['objs'] = {'targs': targs, 'obsts': obsts, 'segs': segs, 'elevators': elevators}

  ret['xBoundary'] = [-3.5, 3.5]
  ret['yBoundary'] = [-3.5, 3.5]

  # radius of an object (so the object doesn't appear as a point)
  ret['radius'] = 0.05

  # step size of the agent movement
  ret['step'] = 0.01

  print segs

  return ret


class ContinuousEnvironment(mdpEnvironment.MDPEnvironment):
  def step(self, state, action, nextState, reward):
    # remove objects if necessary
    # clear this object upon getting it
    loc, orient = state
    nextLoc, nextOrient = nextState

    objInfoLists = self.mdp.getReachedObjects(nextLoc)

    for nextStateType, nextObjId in objInfoLists:
      if nextStateType == 'targs':
        self.mdp.clearObj(nextStateType, nextObjId)
      elif nextStateType == 'segs':
        # be careful with this -
        # once reaching on an segment, deleting the segments before it.
        [self.mdp.clearObj(nextStateType, 0) for i in xrange(nextObjId + 1)]


def getUserAction(state, actionFunction):
  """
  Get an action from the user (rather than the agent).
  
  Used for debugging and lecture demos.
  """
  import graphicsUtils
  action = None
  while True:
    keys = graphicsUtils.wait_for_keys()
    if 'Up' in keys: action = 'north'
    if 'Down' in keys: action = 'south'
    if 'Left' in keys: action = 'west'
    if 'Right' in keys: action = 'east'
    if 'q' in keys: sys.exit(0)
    if action == None: continue
    break
  actions = actionFunction(state)
  if action not in actions:
    action = actions[0]
  return action

def printString(x): print x

def runEpisode(agent, environment, discount, decision, display, message, pause, episode, recorder = None):
  returns = 0
  totalDiscount = 1.0
  environment.reset()
  if 'startEpisode' in dir(agent): agent.startEpisode()
  message("BEGINNING EPISODE: "+str(episode)+"\n")

  runs = 5000

  while True:

    # DISPLAY CURRENT STATE
    state = environment.getCurrentState()
    display(state)
    pause()
    
    # END IF IN A TERMINAL STATE
    actions = environment.getPossibleActions(state)
    runs -= 1
    if environment.isFinal() or runs == 0:
      message("EPISODE "+str(episode)+" COMPLETE: RETURN WAS "+str(returns)+"\n")
      agent.final(state)
      return returns
    
    # GET ACTION (USUALLY FROM AGENT)
    action = decision(state)
    if action == None:
      raise 'Error: Agent returned None action'

    if recorder != None:
      recorder.append((state, action))
    
    # EXECUTE ACTION
    nextState, reward = environment.doAction(action)
    message("Started in state: "+str(state)+
            "\nTook action: "+str(action)+
            "\nEnded in state: "+str(nextState)+
            "\nGot reward: "+str(reward)+"\n")    
    # UPDATE LEARNER
    if 'observeTransition' in dir(agent): 
        agent.observeTransition(state, action, nextState, reward)
    
    environment.step(state, action, nextState, reward)

    returns += reward * totalDiscount
    totalDiscount *= discount

  if 'stopEpisode' in dir(agent):
    agent.stopEpisode()

def parseOptions():
    optParser = optparse.OptionParser()
    optParser.add_option('-d', '--discount',action='store',
                         type='float',dest='discount',default=0.9,
                         help='Discount on future (default %default)')
    optParser.add_option('-r', '--livingReward',action='store',
                         type='float',dest='livingReward',default=0.0,
                         metavar="R", help='Reward for living for a time step (default %default)')
    optParser.add_option('-n', '--noise',action='store',
                         type='float',dest='noise',default=0,
                         metavar="P", help='How often action results in ' +
                         'unintended direction (default %default)' )
    optParser.add_option('-e', '--epsilon',action='store',
                         type='float',dest='epsilon',default=0,
                         metavar="E", help='Chance of taking a random action in q-learning (default %default)')
    optParser.add_option('-l', '--learningRate',action='store',
                         type='float',dest='learningRate',default=0.5,
                         metavar="P", help='TD learning rate (default %default)' )
    optParser.add_option('-i', '--iterations',action='store',
                         type='int',dest='iters',default=10,
                         metavar="K", help='Number of rounds of value iteration (default %default)')
    optParser.add_option('-k', '--episodes',action='store',
                         type='int',dest='episodes',default=1,
                         metavar="K", help='Number of epsiodes of the MDP to run (default %default)')
    optParser.add_option('-g', '--grid',action='store',
                         metavar="G", type='string',dest='grid',default="toy",
                         help='Grid to use (case sensitive; options are BookGrid, BridgeGrid, CliffGrid, MazeGrid, default %default)' )
    optParser.add_option('-w', '--windowSize', metavar="X", type='int',dest='gridSize',default=150,
                         help='Request a window width of X pixels *per grid cell* (default %default)')
    optParser.add_option('-a', '--agent',action='store', metavar="A",
                         type='string',dest='agent',default="random",
                         help='Agent type (options are \'random\', \'value\' and \'q\', default %default)')
    optParser.add_option('-t', '--text',action='store_true',
                         dest='textDisplay',default=False,
                         help='Use text-only ASCII display')
    optParser.add_option('-p', '--pause',action='store_true',
                         dest='pause',default=False,
                         help='Pause GUI after each time step when running the MDP')
    optParser.add_option('-q', '--quiet',action='store_true',
                         dest='quiet',default=False,
                         help='Skip display of any learning episodes')
    optParser.add_option('-s', '--speed',action='store', metavar="S", type=float,
                         dest='speed',default=1.0,
                         help='Speed of animation, S > 1.0 is faster, 0.0 < S < 1.0 is slower (default %default)')
    optParser.add_option('-m', '--manual',action='store_true',
                         dest='manual',default=False,
                         help='Manually control agent')
    optParser.add_option('-v', '--valueSteps',action='store_true' ,default=False,
                         help='Display each step of value iteration')

    opts, args = optParser.parse_args()
    
    if opts.manual and opts.agent != 'q':
      print '## Disabling Agents in Manual Mode (-m) ##'
      opts.agent = None

    # MANAGE CONFLICTS
    if opts.textDisplay or opts.quiet:
    # if opts.quiet:      
      opts.pause = False
      # opts.manual = False
      
    if opts.manual:
      opts.pause = True
      
    return opts

class Plotting:
  def __init__(self, mdp, dim = 800):
    self.mdp = mdp
    self.size = max(mdp.xBoundary[1] - mdp.xBoundary[0], mdp.yBoundary[1] - mdp.yBoundary[0])
    self.radius = mdp.radius / self.size * dim
    self.dim = dim

    def shift(loc):
      """
      shift to the scale of the GraphWin
      """
      return (1.0 * (loc[0] - mdp.xBoundary[0]) / self.size * dim, 1.0 * (loc[1] - mdp.yBoundary[0]) / self.size * dim)

    self.shift = shift
 
  def drawDomain(self):
    """
    Args:
      mdp: parsed from mat file.

    Return:
      win object
    """
    win = GraphWin('Domain', self.dim, self.dim) # give title and dimensions
    win.setBackground('grey')
   
    def drawObjects(label, color):
      """
      Plot the objects as separate dots.
      """
      for obj in self.mdp.objs[label]:
        cir = Circle(Point(self.shift(obj)), self.radius)
        cir.setFill(color)
        cir.draw(win)

    def drawSegments(label, color):
      """
      Plot the adjacent objects as segments.
      """
      prevObj = None
      for obj in self.mdp.objs[label]:
        if prevObj:
          line = Line(Point(self.shift(prevObj)), Point(self.shift(obj)))
          line.setWidth(3)
          line.setFill(color)
          line.draw(win)
        prevObj = obj
      
    pathColor = color_rgb(200, 200, 200)
    drawObjects('targs', 'blue')
    drawObjects('obsts', 'red')
    drawSegments('segs', pathColor)
    drawObjects('elevators', pathColor)

    return win

 
def main():
  opts = parseOptions()

  ###########################
  # GET THE GRIDWORLD
  ###########################

  if opts.grid == 'vr':
    init = loadFromMat('miniRes25.mat', 0)
  elif opts.grid == 'toy':
    init = toyDomain()
  else:
    raise Exception("Unknown environment!")

  mdp = ContinuousWorld(init)
  mdp.setLivingReward(opts.livingReward)
  mdp.setNoise(opts.noise)
  env = ContinuousEnvironment(mdp)

  
  ###########################
  # GET THE DISPLAY ADAPTER
  ###########################
  # FIXME repeated here.
  if not opts.quiet:
    dim = 800
    plotting = Plotting(mdp, dim)
    win = plotting.drawDomain()

  ###########################
  # GET THE AGENT
  ###########################

  # SHOULD BE IMPOSSIBLE TO USE Q OR VALUE ITERATION WITHOUT FUNCTION APPROXIMATION!
  # THE STATE SPACE WOULD BE THE RAW STATE SPACE, WHICH SPANNED BY THE AGENT'S SMALL STEPS!
  import valueIterationAgents, qlearningAgents
  a = None
  if opts.agent == 'value':
    a = valueIterationAgents.ValueIterationAgent(mdp, opts.discount, opts.iters)
  elif opts.agent == 'q':
    continuousEnv = ContinuousEnvironment(mdp)
    actionFn = lambda state: mdp.getPossibleActions(state)
    qLearnOpts = {'gamma': opts.discount, 
                  'alpha': opts.learningRate, 
                  'epsilon': opts.epsilon,
                  'actionFn': actionFn}
    a = qlearningAgents.QLearningAgent(**qLearnOpts)
  elif opts.agent == 'Approximate':
    extractor = featureExtractors.ContinousRadiusLogExtractor(mdp, 'targs')
    continuousEnv = ContinuousEnvironment(mdp)
    actionFn = lambda state: mdp.getPossibleActions(state)
    qLearnOpts = {'gamma': opts.discount, 
                  'alpha': opts.learningRate, 
                  'epsilon': opts.epsilon,
                  'actionFn': actionFn,
                  'extractor': extractor}
    a = qlearningAgents.ApproximateQAgent(**qLearnOpts)
  elif opts.agent == 'Modular':
    import modularAgents
    continuousEnv = ContinuousEnvironment(mdp)
    actionFn = lambda state: mdp.getPossibleActions(state)
    qLearnOpts = {'gamma': opts.discount, 
                  'alpha': opts.learningRate, 
                  'epsilon': opts.epsilon,
                  'actionFn': actionFn}
    a = modularAgents.ModularAgent(**qLearnOpts)
    # here, set the Q tables of the trained modules
    a.setQFuncs(modularAgents.getContinuousWorldFuncs(mdp))
  elif opts.agent == 'random':
    # # No reason to use the random agent without episodes
    if opts.episodes == 0:
      opts.episodes = 10
    class RandomAgent:
      def getAction(self, state):
        return random.choice(mdp.getPossibleActions(state))
      def getValue(self, state):
        return 0.0
      def getQValue(self, state, action):
        return 0.0
      def getPolicy(self, state):
        "NOTE: 'random' is a special policy value; don't use it in your code."
        return 'random'
      def update(self, state, action, nextState, reward):
        pass      
    a = RandomAgent()
  else:
    if not opts.manual: raise 'Unknown agent type: '+opts.agent
    
    
  ###########################
  # RUN EPISODES
  ###########################

  # FIGURE OUT WHAT TO DISPLAY EACH TIME STEP (IF ANYTHING)
  if not opts.quiet:
    def displayCallback(x):
      # display the corresponding state in graphics
      if displayCallback.prevState != None:
        # only draw lines, so ignore the first state
        loc, orient = displayCallback.prevState
        newLoc, newOrient = x

        line = Line(Point(plotting.shift(loc)), Point(plotting.shift(newLoc)))
        line.setWidth(3)
        line.setFill('green')
        line.draw(win)

      displayCallback.prevState = x

    displayCallback.prevState = None
  else:
    displayCallback = lambda x : None

  messageCallback = lambda x: printString(x)
  pauseCallback = lambda : None

  # FIGURE OUT WHETHER THE USER WANTS MANUAL CONTROL (FOR DEBUGGING AND DEMOS)  
  if opts.manual:
    decisionCallback = lambda state : getUserAction(state, mdp.getPossibleActions)
  else:
    decisionCallback = a.getAction  
    
  # RUN EPISODES
  if opts.episodes > 0:
    print
    print "RUNNING", opts.episodes, "EPISODES"
    print
  returns = 0
  for episode in range(1, opts.episodes+1):
    returns += runEpisode(a, env, opts.discount, decisionCallback, displayCallback, messageCallback, pauseCallback, episode)
  if opts.episodes > 0:
    print
    print "AVERAGE RETURNS FROM START STATE: "+str((returns+0.0) / opts.episodes)
    print
    print
    
  if opts.agent == 'Approximate':
    import pprint
    pprint.pprint(a.weights)

  # hold window
  win.getMouse()
  win.close()

if __name__ == '__main__':
  main()
