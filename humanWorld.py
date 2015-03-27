import random
import optparse
import featureExtractors
import continuousWorld
import humanInfoParser
from gridworld import GridworldEnvironment

import numpy as np

from graphics import *
import continuousWorldDomains
import continuousWorldPlot
from numpy import average

class HumanWorld(continuousWorld.ContinuousWorld):
  """
  An MDP that agrees with Matt's human data.

  It is almost the same as ContinuousWorld, but this is in the agent's view.
  The agent uses the distance / angle to the object as state.
  
  The coordinates: vectors (visually) above the x-axis have negative angles.

      -pi/2
  -pi   |
  ------------- 0
   pi   |
       pi/2

  State: (distance, orient) for targ, obst, seg, respectively
  Action: L, R, G
  Transition: same as continuousWorld but 
  Reward: same as continuousWorld.
  """
  # static attributes
  # FIXME overfit
  step = 0.3
  turnAngle = 15.0 / 180 * np.pi
  turnDist = step * 0.25
  walkDist = step * 1

  def __init__(self, init):
    continuousWorld.ContinuousWorld.__init__(self, init)

    self.atBorder = False

  actions = ('L', 'R', 'G')
    
  def getPossibleActions(self, state):
    return HumanWorld.actions

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
  
  @staticmethod
  def transitionSimulate(s, a):
    """
    In some cases, we need to get the next state given the current state and action.
    Because the current state is represented by distance and angle to an object,
    it's a bit tricky to get the distance and angle to the object after taking an action.

    This is done by creating an ad-hoc coordinate space and do a one step simulation.

    Args:
      s: (dist, orient)
      a: action
    Return:
      (newDist, newOrient) after taking a in state s.
    """
    # use human world info for simulation
    # FIXME should put here?
    turnAngle = HumanWorld.turnAngle
    turnDist = HumanWorld.turnDist
    walkDist = HumanWorld.walkDist

    dist, orient = s

    objX = dist * np.cos(orient) 
    objY = dist * np.sin(orient) 
    
    if a == 'G':
      orient = 0
      aX = walkDist; aY = 0
    if a == 'L':
      orient = -turnAngle
      aX = turnDist * np.cos(orient) 
      aY = turnDist * np.sin(orient) 
    elif a == 'R':
      orient = turnAngle
      aX = turnDist * np.cos(orient) 
      aY = turnDist * np.sin(orient) 
    
    # the new state is from (aX, aY) to (objX, objY
    newDist, newOrient = featureExtractors.getDistAngle((aX, aY), (objX, objY), orient)

    return (newDist, newOrient)

# The environment is same as the continuousWorld
HumanEnvironment = continuousWorld.ContinuousEnvironment

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
    runs -= 1
    if environment.isFinal() or runs == 0:
      print "EPISODE "+str(episode)+" COMPLETE: RETURN WAS "+str(returns)+"\n"

      """
      # write number of contact objects into a stat file
      targsNum = len(environment.mdp.collectedTargetSet)
      obstsNum = len(environment.mdp.touchedObstacleSet)
      stats = open('stats','a')
      stats.write(str(targsNum) + ' ' + str(obstsNum) + '\n')
      stats.close()
      """
      deltas = open('deltas','a')
      deltas.write(str(np.average(agent.deltas)) + '\n')
      deltas.close()

      agent.final(state)
      return returns
    
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

    # here, change the environment reponding to the agent's behavior
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
    optParser.add_option('-c', '--category',action='store', metavar="S", type=str,
                         dest='category',default=None,
                         help='For HumanWorld domain to train module separately, can be "targs", "obsts" or "segs"')
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
      opts.pause = False
      # opts.manual = False
      
    if opts.manual:
      opts.pause = True
      
    return opts

def saveValues(values, filename):
  """
  Write to a file in the local directory for temporary use.
  Note that the files in learnedValues/ are used.
  """
  import pickle
  output = open(filename, 'wb')
  pickle.dump(values, output)
  output.close()

def plotQFuncs(values, category):
  """
  Print the values of states in heatmap
  """
  import matplotlib.pyplot as plt

  for act in ['L', 'R', 'G']:
    data = []
    for i in reversed(range(1, 11)): # 1 ~ 10. so that 1 appears at bottom
      row = []
      for j in range(-4, 5): # -4 ~ 4.
        row.append(values[(i, j), act])
      data.append(row)

    plt.imshow(data, interpolation='none')
    plt.xticks(range(9), ['-135', '-90', '-45', '-15', '0', '15', '45', '90', '135'])
    plt.yticks(range(10), ['>10', '10', '5', '4', '3', '2.5', '2', '1.5', '1', '.5'])
    plt.xlabel('Angle');
    plt.ylabel('Distance (x steps)');
    plt.title('Q Table of Module ' + category + ', Action ' + act)
    
    plt.jet()
    plt.colorbar()

    plt.savefig(category + 'Q_' + act + '.png')
    plt.close()
   
def main(): 
  opts = parseOptions()
  possibleCategories = ['targs', 'obsts', 'segs']
  nModules = 4

  ###########################
  # GET THE ENVIRONMENT
  ###########################

  if 'vr' in opts.grid:
    vrDomainId = int(opts.grid[2:])
    init = lambda: continuousWorldDomains.loadFromMat('miniRes25.mat', vrDomainId)

    import pickle
    valueTable = pickle.load(open('learnedValues/values.pkl'))
    values = valueTable[vrDomainId / 8] # 8 domains per task
    
    weights = values[:nModules]
    discounters = None if len(values) == nModules else values[nModules:]

    print "init using domain #", vrDomainId, "with values", weights, "and discounters", discounters
  elif opts.grid == 'vrTrain':
    init = lambda: continuousWorldDomains.loadFromMat('miniRes25.mat', 0, randInit = True)
  elif opts.grid == 'toy':
    if not opts.category in possibleCategories:
      raise Exception('Unexpected category ' + opts.category)
    init = lambda: continuousWorldDomains.toyDomain(opts.category)
  elif opts.grid == 'simple':
    if not opts.category in possibleCategories:
      raise Exception('Unexpected category ' + opts.category)
    init = lambda: continuousWorldDomains.simpleToyDomain(opts.category)
  else:
    raise Exception("Unknown environment!")

  mdp = HumanWorld(init())
  mdp.setLivingReward(opts.livingReward)
  mdp.setNoise(opts.noise)
  env = HumanEnvironment(mdp)

  # plot the environment
  if not opts.quiet:
    dim = 800
    plotting = continuousWorldPlot.Plotting(mdp, dim)
    # draw the environment -- path, targets, etc.
    win = plotting.drawDomain()
    # draw human trajectories
    if 'vr' in opts.grid:
      humanInfoParser.plotHuman(plotting, win, range(25, 29), vrDomainId)

  # get the learning agent
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
    a.setValues('learnedValues/humanAgent' + opts.category + 'Values.pkl')
    a.setStateFilter(featureExtractors.getHumanViewBins(mdp, opts.category))
  elif opts.agent == 'sarsa':
    import sarsaLambdaAgents
    gridWorldEnv = GridworldEnvironment(mdp)
    actionFn = lambda state: mdp.getPossibleActions(state)
    qLearnOpts = {'gamma': opts.discount, 
                  'alpha': opts.learningRate, 
                  'epsilon': opts.epsilon,
                  'actionFn': actionFn}
    a = sarsaLambdaAgents.SarsaLambdaAgent(**qLearnOpts)
  elif opts.agent == 'Approximate':
    extractor = featureExtractors.HumanViewExtractor(mdp, opts.category)
    actionFn = lambda state: mdp.getPossibleActions(state)
    qLearnOpts = {'gamma': opts.discount, 
                  'alpha': opts.learningRate, 
                  'epsilon': opts.epsilon,
                  'actionFn': actionFn,
                  'extractor': extractor}
    a = qlearningAgents.ApproximateVAgent(**qLearnOpts)
    a.setWeights('learnedValues/humanAgent' + opts.category + 'Weights.pkl')
  elif 'Modular' in opts.agent:
    # for modular agents
    import modularAgents, modularQFuncs
    actionFn = lambda state: mdp.getPossibleActions(state)
    qLearnOpts = {'gamma': opts.discount, 
                  'alpha': opts.learningRate, 
                  'epsilon': opts.epsilon,
                  'actionFn': actionFn}
    a = modularAgents.ReducedModularAgent(**qLearnOpts)
    a.setWeights(weights)
    #a.setWeights([0.2, 0, 1]) # TEST

    if opts.agent == 'Modular' or opts.agent == 'ModularQ':
      # way 1: using q tables
      a.setStateFilter(featureExtractors.getHumanDiscreteState(mdp))
      a.setQFuncs(modularQFuncs.getHumanWorldDiscreteFuncs())
    elif opts.agent == 'ModularV':
      # way 2: using q functions
      a.setStateFilter(featureExtractors.getHumanContinuousState(mdp))
      a.setQFuncs(modularQFuncs.getHumanWorldQPotentialFuncs(discounters))
      #a.setQFuncs(modularQFuncs.getHumanWorldQPotentialFuncs()) # TEST
    else:
      raise Exception("Unknown modular agent.")
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
    
  # FIGURE OUT WHAT TO DISPLAY EACH TIME STEP (IF ANYTHING)
  if not opts.quiet:
    displayCallback = plotting.plotHumanPath
  else:
    displayCallback = lambda x: None

  if not opts.quiet:
    messageCallback = lambda x: printString(x)
  else:
    messageCallback = lambda x: None

  if opts.pause:
    pauseCallback = lambda : raw_input("Press enter to continue.")
  else:
    pauseCallback = lambda : None

  # FIGURE OUT WHETHER THE USER WANTS MANUAL CONTROL (FOR DEBUGGING AND DEMOS)  
  if opts.manual:
    decisionCallback = lambda state : continuousWorld.getUserAction(state, mdp.getPossibleActions)
  else:
    decisionCallback = a.getAction  
    
  # RUN EPISODES
  if opts.episodes > 0:
    print
    print "RUNNING", opts.episodes, "EPISODES"
    print
  returns = 0
  for episode in range(1, opts.episodes+1):
    # Some environments have random settings (random init state, etc.).
    # So reset the environment every time.
    mdp.__init__(init()) 

    returns += runEpisode(a, env, opts.discount, decisionCallback, displayCallback, messageCallback, pauseCallback, episode)
  if opts.episodes > 0:
    print
    print "AVERAGE RETURNS FROM START STATE: "+str((returns+0.0) / opts.episodes)
    print
    print
    
  if opts.agent == 'Approximate':
    print a.weights
    saveValues(a.weights, 'humanAgent' + opts.category + 'Weights.pkl')
  elif opts.agent == 'q':
    # output learned values to pickle file
    saveValues(a.values, 'humanAgent' + opts.category + 'Values.pkl')
    plotQFuncs(a.values, opts.category)

  # hold window
  if not opts.quiet and 'vr' in opts.grid:
    handler = 'task_' + str(vrDomainId / 8 + 1) + '_room_' + str(vrDomainId + 1)
    win.postscript(file=handler+".eps")
    win.close()

if __name__ == '__main__':
  main()
