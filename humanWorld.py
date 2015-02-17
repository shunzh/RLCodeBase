import random
import sys
import optparse
import featureExtractors
import continuousWorld
import humanInfoParser
from gridworld import GridworldEnvironment
import sarsaLambdaAgents

import numpy as np

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

    self.turnAngle = humanInfoParser.turnAngle
    self.turnDist = self.step * 0.25
    self.walkDist = self.step * 1
    self.atBorder = False

  actions = ('L', 'R', 'G')
    
  def getPossibleActions(self, state):
    """
    L: Turn left 30 degrees and walk ahead 0.05m.
    R: Turn right 30 degrees and walk ahead 0.05m.
    G: Go ahead 0.2m.
    """
    return HumanWorld.actions

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
    if display: display.drawPath(state)
    pause()
    
    # END IF IN A TERMINAL STATE
    runs -= 1
    if environment.isFinal() or runs == 0:
      print "EPISODE "+str(episode)+" COMPLETE: RETURN WAS "+str(returns)+"\n"

      """
      # write number of contact objects into a stat file
      targsNum = len(environment.mdp.collectedTargetSet['targs'])
      obstsNum = len(environment.mdp.touchedObstacleSet)
      stats = open('stats','a')
      stats.write(str(targsNum) + ' ' + str(obstsNum) + '\n')
      stats.close()
      """

      # mark touched objects here
      if display: display.highlight(environment.mdp)

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

    # here, change the environment reponding to the agent's behavior
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

def plotQFuncs(values, filename):
  """
  Print the values of states in heatmap
  """
  import matplotlib.pyplot as plt

  data = []
  for i in reversed(range(1, 11)): # 1 ~ 10. so that 1 appears at bottom
    row = []
    for j in range(-4, 5): # -4 ~ 4.
      row.append(max([values[(i, j), act] for act in ['L', 'R', 'G']]))
    data.append(row)

  plt.imshow(data, interpolation='none')
  plt.xticks(range(9), ['-135', '-90', '-45', '-15', '0', '15', '45', '90', '135'])
  plt.yticks(range(10), ['>10', '10', '5', '4', '3', '2.5', '2', '1.5', '1', '.5'])
  plt.xlabel('Angle');
  plt.ylabel('Distance (x steps)');

  plt.jet()
  plt.colorbar()

  plt.savefig(filename)
   
def main(): 
  opts = parseOptions()
  possibleCategories = ['targs', 'obsts', 'segs']

  ###########################
  # GET THE ENVIRONMENT
  ###########################

  if 'vr' in opts.grid:
    vrDomainId = int(opts.grid[2:])
    init = lambda: continuousWorld.loadFromMat('miniRes25.mat', vrDomainId)

    import pickle
    weightTable = pickle.load(open('learnedValues/weights.pkl'))
    weights = weightTable[vrDomainId / 8] # 8 domains per task

    print "init using domain #", vrDomainId, "with weights", weights
  elif opts.grid == 'vrTrain':
    init = lambda: continuousWorld.loadFromMat('miniRes25.mat', 0, randInit = True)
  elif opts.grid == 'toy':
    if not opts.category in possibleCategories:
      raise Exception('Unexpected category ' + opts.category)
    init = lambda: continuousWorld.toyDomain(opts.category)
  elif opts.grid == 'simple':
    if not opts.category in possibleCategories:
      raise Exception('Unexpected category ' + opts.category)
    init = lambda: continuousWorld.simpleToyDomain(opts.category)
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
    # draw the environment -- path, targets, etc.
    win = plotting.drawDomain()
    # draw human trajectories
    if 'vr' in opts.grid:
      humanInfoParser.plotHuman(plotting, win, range(25, 29), vrDomainId)

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
    a.setValues('learnedValues/humanAgent' + opts.category + 'Values.pkl')
    a.setStateFilter(featureExtractors.getHumanViewBins(mdp, opts.category))
  elif opts.agent == 'sarsa':
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
  elif opts.agent == 'Modular':
    import modularAgents, modularQFuncs
    actionFn = lambda state: mdp.getPossibleActions(state)
    qLearnOpts = {'gamma': opts.discount, 
                  'alpha': opts.learningRate, 
                  'epsilon': opts.epsilon,
                  'actionFn': actionFn}
    a = modularAgents.ReducedModularAgent(**qLearnOpts)
    a.setWeights(weights)
    #a.setWeights([0, 0, 1]) #DEBUG
    a.setStateFilter(featureExtractors.getHumanContinuousState(mdp))
    a.setQFuncs(modularQFuncs.getHumanWorldQPotentialFuncs())
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
    class DisplayCallback:
      """
      The class that plots the behavior of the agent on the fly.
      """
      def __init__(self):
        self.prevState = None

      def drawPath(self, x):
        """
        display the corresponding state in graphics
        """
        if self.prevState != None:
          # only draw lines, so ignore the first state
          loc, orient = self.prevState
          newLoc, orient = x

          line = Line(Point(plotting.shift(loc)), Point(plotting.shift(newLoc)))
          line.setWidth(5)
          line.setFill(color_rgb(0, 255, 0))
          line.draw(win)

        self.prevState = x

      def highlight(self, mdp):
        """
        highlight the elements in the list l
        elements of l are (type, id)
        """
        for loc in mdp.collectedTargetSet + mdp.touchedObstacleSet:
          cir = Circle(Point(plotting.shift(loc)), 7)
          cir.setFill(color_rgb(255, 0, 0))
          cir.draw(win)

    displayCallback = DisplayCallback()
  else:
    displayCallback = None

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
    plotQFuncs(a.values, 'humanAgent' + opts.category + 'Q.png')

  # hold window
  if not opts.quiet and 'vr' in opts.grid:
    handler = 'task_' + str(vrDomainId / 8 + 1) + '_room_' + str(vrDomainId + 1)
    win.postscript(file=handler+".eps")
    win.close()

if __name__ == '__main__':
  main()
