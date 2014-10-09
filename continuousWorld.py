import random
import sys
import mdp
import environment
import util
import optparse

import numpy
import numpy.linalg
import warnings

from game import Actions

class ContinuousWorld(mdp.MarkovDecisionProcess):
  """
  A MDP that captures continuous state space, while the agent moves in discrete steps.

  State: (location, last visited segment)
  Action: 8 directional movement, with a fixed step size.
  Transition: trivial.
  Reward: upon reaching a target / obstacle, obtain the corresponding reward.
          upon reaching a segment, if this segment is newer than the one visited before (encoded in the state space),
          then obtain the reward.
  """
  def __init__(self):
    self.loadFromMat('miniRes25.mat', 0)

    # reward values that getReward will use
    self.rewards = {'targs': 1, 'obsts': -1, 'segs': 0.1, 'start': 0, 'end': 0}

    # parameters
    self.livingReward = 0.0
    self.noise = 0.0 # DUMMY - need assumption on what it means to be noisy

  def loadFromMat(self, filename, domainId):
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
    s['newRes']['all_objs']

    numObj = len(s['newRes']['all_objs']['id'])

    targs = []
    obsts = []
    segs = []
    elevators = []

    xMin = numpy.inf, xMax = -numpy.inf
    yMin = numpy.inf, yMax = -numpy.inf

    for idx in xrange(numObj):
      name = s['newRes']['all_objs']['id'][idx]

      x = s['newRes']['all_objs']['object_location']['x'][domainId][idx]
      y = s['newRes']['all_objs']['object_location']['y'][domainId][idx]

      if x < xMin: xMin = x
      if x > xMax: xMax = x
      if y < yMin: yMin = y
      if y > yMax: yMax = y

      if 'targ' in name:
        targs.append((x, y))
      elif 'obst' in name:
        obsts.append((x, y))
      elif 'seg' in name:
        segs.append((x, y))
      elif 'elevator' in name:
        elevators.append((x, y))
      else:
        warnings.warn("Dropped unkown object typed '" + name + "' indexed at " + str(idx))

    if len(elevators) < 2:
      raise Exception("Elevators cannot be undefined.")

    self.objs = {'targs': targs, 'obsts': obsts, 'segs': segs, 'elevators': elevators}

    # TODO add buffer?
    self.xBoundary = [xMin, xMax]
    self.yBoundary = [yMin, yMax]

    # radius of an object (so the object doesn't appear as a point)
    self.radius = 0.2

    # step size of the agent movement
    self.step = 0.1


  def closeToAnObject(self, l):
    """
    Determine whether a state is close to any object, within radius.
    Args:
      l: the loc to be checked.
    Return:
      String, int: the type of object that it's in the radius of, and its id.
                   It is (None, None) if it's close to nothing (so easy to be parsed)

    #FIXME not checked whether it's close to multiple objects. However, this won't happen in a valid domain.
    """
    for key, locs in self.objs.items():
      for idx in xrange(len(locs)):
        dist = numpy.linalg.norm(numpy.subtract(l, locs[idx]))
        if dist < self.radius:
          return (key, idx)

    # if it's close to nothing
    return (None, None)

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
    Return list of discrete states. This is usually for sanity check.
    """
    raise Exception("getStates: not implemented for continuous domains.")
    return None
        
  def getReward(self, state, action, nextState):
    """
    Get reward for state, action, nextState transition.
    
    Note that the reward depends only on the state being
    departed (as in the R+N book examples, which more or
    less use this convention).
    """
    loc, seg = state
    nextLoc, nextSeg = nextState
    
    if seg != nextSeg:
      # reaching a new segment
      return self.rewards["segs"]
    else:
      # check whether reaching a target or obstacle
      stateType, objId = closeToAnObject(self, loc)
      nextStateType, newObjId = closeToAnObject(self, nextLoc)

      if stateType != nextStateType and nextStateType != None:
        # only consider when the types of states change
        # i.e. moving from target to target doesn't give extra reward.
        return self.rewards[nextStateType]
        
  def getStartState(self):
    """
    Start at the starting location, with no segment previously visited.
    """
    return (self.objs['elevators'][0], 0)
    
  def isFinal(self, state):
    """
    Check whether we should terminate at this state.
    """
    loc, seg = state
    return self.closeToAnObject(loc) == ('elevators', 1)
                   
  def getTransitionStatesAndProbs(self, state, action):
    """
    Basically following physics laws, but considering:
    - bound back within self.xBoundary and self.yBoundary
    - change seg in the state representation upon reaching a new segment
    """
    if self.isFinal(state):
      return []
    
    loc, seg = state

    # move to new loc and check whether it's allowed
    newLoc = numpy.add(loc, Actions._directions[action])
    newLoc = (self.__isAllowed(newLoc) and newLoc) or loc
    
    stateType, objId = closeToAnObject(self, newLoc)
    if stateType == 'segs' and objId > seg:
      newSeg = objId
    else:
      newSeg = seg

    successors = [((newLoc, newSeg), 1)]

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
        
  def __isAllowed(self, loc):
    """
    Check whether this state is valid
    """
    x, y = loc
    if x < self.xBoundary[0] or x >= self.xBoundary[1]: return False
    if y < self.yBoundary[0] or y >= self.yBoundary[1]: return False

class ContinuousEnvironment(environment.Environment):
  """
  which holds a mdp object.

  #FIXME this is essentially the same for all the domains, consider abstract this.
  """
  def __init__(self, mdp):
    self.mdp = mdp
    self.reset()
            
  def getCurrentState(self):
    return self.state
        
  def getPossibleActions(self, state):        
    return self.mdp.getPossibleActions(state)
        
  def doAction(self, action):
    successors = self.mdp.getTransitionStatesAndProbs(self.state, action) 
    sum = 0.0
    rand = random.random()
    state = self.getCurrentState()
    for nextState, prob in successors:
      sum += prob
      if sum > 1.0:
        raise 'Total transition probability more than one; sample failure.' 
      if rand < sum:
        reward = self.mdp.getReward(state, action, nextState)
        self.state = nextState
        return (nextState, reward)
    raise 'Total transition probability less than one; sample failure.'    
        
  def reset(self):
    self.state = self.mdp.getStartState()

  def isFinal(self):
    return self.mdp.isFinal(self.state)

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

def runEpisode(agent, environment, discount, decision, display, message, pause, episode):
  returns = 0
  totalDiscount = 1.0
  environment.reset()
  if 'startEpisode' in dir(agent): agent.startEpisode()
  message("BEGINNING EPISODE: "+str(episode)+"\n")

  runs = 500

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
    
    # EXECUTE ACTION
    nextState, reward = environment.doAction(action)
    message("Started in state: "+str(state)+
            "\nTook action: "+str(action)+
            "\nEnded in state: "+str(nextState)+
            "\nGot reward: "+str(reward)+"\n")    
    # UPDATE LEARNER
    if 'observeTransition' in dir(agent): 
        agent.observeTransition(state, action, nextState, reward)
    
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
                         type='float',dest='epsilon',default=0.3,
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
                         metavar="G", type='string',dest='grid',default="BookGrid",
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

  
if __name__ == '__main__':
  
  opts = parseOptions()

  ###########################
  # GET THE GRIDWORLD
  ###########################

  mdp = ContinuousWorld()
  mdp.setLivingReward(opts.livingReward)
  mdp.setNoise(opts.noise)
  env = ContinuousEnvironment(mdp)

  
  ###########################
  # GET THE DISPLAY ADAPTER
  ###########################
  # TODO

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
    a.setQFuncs(modularAgents.getObsAvoidFuncs(mdp))
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
  # DISPLAY Q/V VALUES BEFORE SIMULATION OF EPISODES
  

  # FIGURE OUT WHAT TO DISPLAY EACH TIME STEP (IF ANYTHING)
  displayCallback = lambda x: None
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
    
  # DISPLAY POST-LEARNING VALUES / Q-VALUES
  if opts.agent == 'q' or opts.agent == 'Approximate' or opts.agent == 'Modular' and not opts.manual:
    messageCallback("Q-VALUES AFTER "+str(opts.episodes)+" EPISODES")
    messageCallback("VALUES AFTER "+str(opts.episodes)+" EPISODES")
