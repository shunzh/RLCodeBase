import random
import sys
import mdp
import environment
import util
import optparse
import featureExtractors
import continuousWorld

import numpy as np
import numpy.linalg
import warnings

from graphics import *

class HumanWorld(continuousWorld.ContinuousWorld):
  """
  An MDP that agrees with Matt's human data.

  It is almost the same as ContinuousWorld, but this is in the agent's view.
  The agent stays in (0, 0) forever, while the distance / angle to the object changes.

  State: (location, ref to obj)
  Action: L, R, G
  Transition: same as continuousWorld but 
  Reward: same as continuousWorld.
  """
  def __init__(self, init):
    continuousWorld.ContinuousWorld.__init__(self, init)

    self.turnAngle = 30.0 / 180 * np.pi
    self.slightTurnAngle = 15.0 / 180 * np.pi
    # this is scaled by the step size in the domain
    self.turnDist = self.step * 0.25
    self.walkDist = self.step * 1
    self.atBorder = False
    
  def getPossibleActions(self, state):
    """
    L: Turn left 30 degrees and walk ahead 0.05m.
    R: Turn right 30 degrees and walk ahead 0.05m.
    G: Go ahead 0.2m.
    """
    return ('L', 'R', 'G')

  def getReward(self, state, action, nextState):
    reward = continuousWorld.ContinuousWorld.getReward(self, state, action, nextState)

    if self.atBorder and 'borderReward' in self.__dict__.keys():
      reward += self.borderReward

    return reward

  def isFinal(self, state):
    return continuousWorld.ContinuousWorld.isFinal(self, state) or self.atBorder

  def getTransitionStatesAndProbs(self, state, action):
    """
    Responde to actions.

    Use: self.[turnAngle, turnDist, walkDist]
    """
    loc, orient = state

    if action == 'L':
      newOrient = orient - self.turnAngle
      d = self.turnDist
    elif action == 'R':
      newOrient = orient + self.turnAngle
      d = self.turnDist
    elif action == 'G':
      newOrient = orient
      d = self.walkDist
    else:
      raise Exception("Unknown action.")

    newOrient = featureExtractors.adjustAngle(newOrient)

    dv = (d * np.cos(newOrient), d * np.sin(newOrient))
    newLoc = np.add(loc, dv)
    if not self.isAllowed(newLoc):
      self.atBorder = True
      newLoc = loc
    else:
      self.atBorder = False

    newLoc = tuple(newLoc) # make sure type is consistent

    newState = (newLoc, newOrient)

    return [(newState, 1)]


#Environment for human.
HumanEnvironment = continuousWorld.ContinuousEnvironment


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
      print "EPISODE "+str(episode)+" COMPLETE: RETURN WAS "+str(returns)+"\n"
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
    if 'getState' in dir(agent):
      message("Started in belief state: " + str(agent.getState(state)) + 
              "\nEnded in belief state: " + str(agent.getState(nextState)) + "\n")

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
                         type='float',dest='discount',default=0.2,
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

def parseValues(values, filename):
  # do not update everytime
  import pickle
  output = open(filename, 'wb')
  pickle.dump(values, output)
  output.close()
   
def main(): 
  opts = parseOptions()

  ###########################
  # GET THE GRIDWORLD
  ###########################

  category = 'targs'
  #category = 'obsts'
  #category = 'segs'

  if category == 'segs': trainCategory = 'targs'
  else: trainCategory = category

  if opts.grid == 'vr':
    init = lambda: continuousWorld.loadFromMat('miniRes25.mat', 24)
  elif opts.grid == 'toy':
    init = lambda: continuousWorld.toyDomain(category)
  elif opts.grid == 'simple':
    init = lambda: continuousWorld.simpleToyDomain(category)
  else:
    raise Exception("Unknown environment!")

  mdp = HumanWorld(init())
  mdp.setLivingReward(opts.livingReward)
  mdp.setNoise(opts.noise)
  env = HumanEnvironment(mdp)

  
  ###########################
  # GET THE DISPLAY ADAPTER
  ###########################

  if not opts.quiet:
    dim = 800
    plotting = continuousWorld.Plotting(mdp, dim)
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
    continuousEnv = HumanEnvironment(mdp)
    actionFn = lambda state: mdp.getPossibleActions(state)
    qLearnOpts = {'gamma': opts.discount, 
                  'alpha': opts.learningRate, 
                  'epsilon': opts.epsilon,
                  'actionFn': actionFn}
    a = qlearningAgents.ReducedQLearningAgent(**qLearnOpts)
    a.setValues('learnedValues/humanAgent' + trainCategory + 'Values.pkl')
    a.setStateFilter(featureExtractors.getHumanViewBins(mdp, category))
    a.setLambdaValue(0.1)
  elif opts.agent == 'sarsa':
    gridWorldEnv = GridworldEnvironment(mdp)
    actionFn = lambda state: mdp.getPossibleActions(state)
    qLearnOpts = {'gamma': opts.discount, 
                  'alpha': opts.learningRate, 
                  'epsilon': opts.epsilon,
                  'actionFn': actionFn}
    a = sarsaLambdaAgents.SarsaLambdaAgent(**qLearnOpts)
  elif opts.agent == 'Approximate':
    extractor = featureExtractors.HumanViewExtractor(mdp, category)
    continuousEnv = HumanEnvironment(mdp)
    actionFn = lambda state: mdp.getPossibleActions(state)
    qLearnOpts = {'gamma': opts.discount, 
                  'alpha': opts.learningRate, 
                  'epsilon': opts.epsilon,
                  'actionFn': actionFn,
                  'extractor': extractor}
    a = qlearningAgents.ApproximateVAgent(**qLearnOpts)
    a.setWeights('learnedValues/humanAgent' + trainCategory + 'Weights.pkl')
  elif opts.agent == 'Modular':
    import modularAgents
    continuousEnv = HumanEnvironment(mdp)
    actionFn = lambda state: mdp.getPossibleActions(state)
    qLearnOpts = {'gamma': opts.discount, 
                  'alpha': opts.learningRate, 
                  'epsilon': opts.epsilon,
                  'actionFn': actionFn}
    a = modularAgents.ReducedModularAgent(**qLearnOpts)
    #a.setStateFilter(featureExtractors.getHumanContinuousState(mdp))
    #a.setQFuncs(modularAgents.getHumanWorldContinuousFuncs())
    a.setStateFilter(featureExtractors.getHumanDiscreteState(mdp))
    a.setQFuncs(modularAgents.getHumanWorldDiscreteFuncs())
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
        newLoc, orient = x

        line = Line(Point(plotting.shift(loc)), Point(plotting.shift(newLoc)))
        line.setWidth(5)
        line.setFill(color_rgb(0, 255, 0))
        line.draw(win)

      displayCallback.prevState = x

    displayCallback.prevState = None
  else:
    displayCallback = lambda x: None

  if not opts.quiet:
    messageCallback = lambda x: printString(x)
  else:
    messageCallback = lambda x: None

  pauseCallback = lambda : None
  #pauseCallback = lambda : raw_input("waiting")

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
    mdp.__init__(init()) # reset the environment every time.
    returns += runEpisode(a, env, opts.discount, decisionCallback, displayCallback, messageCallback, pauseCallback, episode)
  if opts.episodes > 0:
    print
    print "AVERAGE RETURNS FROM START STATE: "+str((returns+0.0) / opts.episodes)
    print
    print
    
  if opts.agent == 'Approximate':
    print a.weights
    parseValues(a.weights, 'humanAgent' + category + 'Weights.pkl')
  elif opts.agent == 'q':
    parseValues(a.values, 'humanAgent' + category + 'Values.pkl')

  # hold window
  if not opts.quiet:
    win.getMouse()
    win.close()

if __name__ == '__main__':
  main()
