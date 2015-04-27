import optparse
from continuousWorld import ContinuousWorld, ContinuousEnvironment
import featureExtractors
import sys

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
    optParser.add_option('-t', '--eligibilityTraces',action='store',
                         type='float',dest='lmd',default=0.0,
                         metavar="R", help='Eligibility traces (default %default)')
    optParser.add_option('-n', '--noise',action='store',
                         type='float',dest='noise',default=0,
                         metavar="P", help='How often action results in ' +
                         'unintended direction (default %default)' )
    optParser.add_option('-e', '--epsilon',action='store',
                         type='float',dest='epsilon',default=0,
                         metavar="E", help='Chance of taking a random action in q-learning (default %default)')
    optParser.add_option('-l', '--learningRate',action='store',
                         type='float',dest='learningRate',default=0.1,
                         metavar="P", help='TD learning rate (default %default)' )
    optParser.add_option('-i', '--iterations',action='store',
                         type='int',dest='iters',default=10,
                         metavar="K", help='Number of rounds of value iteration (default %default)')
    optParser.add_option('-k', '--episodes',action='store',
                         type='int',dest='episodes',default=1,
                         metavar="K", help='Number of epsiodes of the MDP to run (default %default)')
    optParser.add_option('-g', '--grid',action='store',
                         metavar="G", type='string',dest='grid',default="toy",
                         help='Specify the training domain. Can be \'simple\', \'toy\', \'vr{number}\'')
    optParser.add_option('-w', '--windowSize', metavar="X", type='int',dest='gridSize',default=150,
                         help='Request a window width of X pixels *per grid cell* (default %default)')
    optParser.add_option('-a', '--agent',action='store', metavar="A",
                         type='string',dest='agent',default="random",
                         help='Agent type (options are \'random\', \'value\' and \'q\', default %default)')
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
    optParser.add_option('-c', '--category',action='store', metavar="S", type=str,
                         dest='category',default=None,
                         help='For HumanWorld domain to train module separately, can be "targs", "obsts" or "segs"')

    opts, args = optParser.parse_args()
    
    if opts.manual and opts.agent != 'q':
      print '## Disabling Agents in Manual Mode (-m) ##'
      opts.agent = None

    # MANAGE CONFLICTS
    if opts.quiet:
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
    import continuousWorldPlot
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
    import baselineAgents
    a = baselineAgents.RandomAgent()
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
