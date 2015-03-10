import random
import sys
import mdp
import mdpEnvironment
import util
import optparse
import featureExtractors

import numpy as np
import numpy.linalg

from game import Actions
import continuousWorldPlot

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
    self.rewards = {'targs': 1, 'obsts': -1, 'segs': 1, 'elevators': 0, 'entrance': 0}
    self.noise = 0.0 # DUMMY - need assumption on what it means to be noisy

    # stat set
    self.touchedObstacleSet = []
    self.collectedTargetSet = []

    if not 'livingReward' in self.__dict__.keys():
      self.livingReward = 0

  def getReachedObjects(self, l):
    """
    Determine whether a state is close to any object, within radius.
    The radius for obstacles should be larger than that of targets.
    A waypoint is 'reached' when the agent gets closer to the next waypoint.

    Args:
      l: the loc to be checked.
    Return:
      [(String, int)]: A list of the type of object that it's in the radius of, and its id.
                       Return [] if nothing matches.
    """
    ret = []

    # get a target?
    tLocs = self.objs['targs']
    for targIdx in xrange(len(tLocs)):
      dist = numpy.linalg.norm(np.subtract(l, tLocs[targIdx]))
      if dist < self.radius:
        ret.append(('targs', targIdx))

    # run into an obstacle?
    oLocs = self.objs['obsts']
    for obstIdx in xrange(len(tLocs)):
      dist = numpy.linalg.norm(np.subtract(l, oLocs[obstIdx]))
      if dist < self.radius:
        ret.append(('obsts', obstIdx))

    # close to the next segment?
    sLocs = self.objs['segs']
    segIdx = 0
    """
    while segIdx < len(sLocs):
      dist = numpy.linalg.norm(np.subtract(l, sLocs[segIdx]))
      if dist < self.radius * 2: # larger buffer
        ret.append(('segs', segIdx))
        segIdx += 1
      else:
        break
    """
    # rubber band
    if len(sLocs) > segIdx + 1:
      # when get closer to the next one
      distSeg1 = numpy.linalg.norm(np.subtract(l, sLocs[segIdx]))
      distSeg2 = numpy.linalg.norm(np.subtract(l, sLocs[segIdx + 1]))
      print distSeg1, distSeg2
      if distSeg1 > distSeg2 :
        ret.append(('segs', segIdx))
        segIdx += 1
    elif len(sLocs) == 1:
      # if only one left, just approach it
      distSeg = numpy.linalg.norm(np.subtract(l, sLocs[segIdx]))
      if distSeg < self.radius * 2: # larger buffer
        ret.append(('segs', 0))

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
    nextLoc, nextOrient = nextState
    
    reward = 0

    # get the list of contacted objects
    objInfoList = self.getReachedObjects(nextLoc)

    for nextStateType, nextObjId in objInfoList:
      if nextStateType != 'segs':
        # add rewards for target and obstacles
        reward += self.rewards[nextStateType]

        # keep a set of reached objects
        objLoc = self.objs[nextStateType][nextObjId]
        if nextStateType == 'obsts' and not objLoc in self.touchedObstacleSet:
          self.touchedObstacleSet.append(objLoc)
        elif nextStateType == 'targs':
          self.collectedTargetSet.append(objLoc)

    # give rewards for waypoint segments, except training targets or obstacles
    if not hasattr(self, 'category') or self.category == 'segs': 
      dist = featureExtractors.getClosestObj(loc, self.objs['segs'])
      nextDist = featureExtractors.getClosestObj(nextLoc, self.objs['segs'])
      if nextDist < dist:
        # reward for shrinking distance
        reward += self.rewards['segs']

    return reward or self.livingReward
      
  def clearObj(self, objType, objId):
    """
    Clear an object from self.objs, usually because the agent has got it.
    """
    del self.objs[objType][objId]

  def getStartState(self):
    """
    Start at the starting location, with no segment previously visited.
    """
    loc = self.objs['entrance']

    # face towards the center of the domain
    if loc[0] < 0: angle = 45.0 / 180 * np.pi
    else: angle = - 135.0 / 180 * np.pi
    return (loc, angle)
    
  def isFinal(self, state):
    """
    Check whether we should terminate at this state.

    Condition: reached exit elevator or no target left.
    """
    loc, orient = state

    #return len(self.objs['segs']) == 0 or len(self.objs['targs']) == 0
    return len(self.objs['segs']) == 0

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

class ContinuousEnvironment(mdpEnvironment.MDPEnvironment):
  def step(self, state, action, nextState, reward):
    # remove objects if necessary
    # clear this object upon getting it
    loc, orient = state
    nextLoc, nextOrient = nextState

    objInfoLists = self.mdp.getReachedObjects(nextLoc)
    # to remove object by id, make sure remove the ones with larger id first
    # so the list won't screw up.
    if len(objInfoLists) > 0: objInfoLists.reverse()

    for nextStateType, nextObjId in objInfoLists:
      if nextStateType == 'targs' or nextStateType == 'segs':
        print nextStateType, self.mdp.objs[nextStateType][nextObjId]
        self.mdp.clearObj(nextStateType, nextObjId)


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
                         type='float',dest='discount',default=0.5,
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


def main():
  opts = parseOptions()

  ###########################
  # GET THE GRIDWORLD
  ###########################

  import continuousWorldDomains
  if opts.grid == 'vr':
    init = continuousWorldDomains.loadFromMat('miniRes25.mat', 0)
  elif opts.grid == 'toy':
    init = continuousWorldDomains.toyDomain()
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
    plotting = continuousWorldPlot.Plotting(mdp, dim)
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
    import modularAgents, modularQFuncs
    continuousEnv = ContinuousEnvironment(mdp)
    actionFn = lambda state: mdp.getPossibleActions(state)
    qLearnOpts = {'gamma': opts.discount, 
                  'alpha': opts.learningRate, 
                  'epsilon': opts.epsilon,
                  'actionFn': actionFn}
    a = modularAgents.ModularAgent(**qLearnOpts)
    # here, set the Q tables of the trained modules
    a.setQFuncs(modularQFuncs.getContinuousWorldFuncs(mdp))
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
    displayCallback = plotting.plotHumanPath
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
    
  # hold window
  win.getMouse()
  win.close()

if __name__ == '__main__':
  main()
