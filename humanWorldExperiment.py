import numpy as np
import continuousWorldDomains
from humanWorld import HumanWorld, HumanEnvironment
import continuousWorldPlot
import humanInfoParser
import featureExtractors
import continuousWorldExperiment
from graphics import *

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
      """

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

parseOptions = continuousWorldExperiment.parseOptions

def saveValues(values, filename):
  """
  Write to a file in the local directory for temporary use.
  Note that the files in learnedValues/ are used.
  """
  import pickle
  output = open(filename, 'wb')
  pickle.dump(values, output)
  output.close()
  
def main(): 
  opts = parseOptions()
  possibleCategories = ['targs', 'obsts', 'segs']
  nModules = 3

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
    discounters = [] if len(values) == nModules else values[nModules:]

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

  a = None
  if opts.agent == 'value':
    import valueIterationAgents
    a = valueIterationAgents.ValueIterationAgent(mdp, opts.discount, opts.iters)
  elif opts.agent == 'q':
    import qlearningAgents
    actionFn = lambda state: mdp.getPossibleActions(state)
    qLearnOpts = {'gamma': opts.discount, 
                  'alpha': opts.learningRate, 
                  'epsilon': opts.epsilon,
                  'actionFn': actionFn}
    a = qlearningAgents.ReducedQLearningAgent(**qLearnOpts)
    a.setValues('learnedValues/humanAgent' + opts.category + 'Values.pkl')
    a.setStateFilter(featureExtractors.getHumanViewBins(mdp, opts.category))
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
    import modularAgents, modularQFuncs, config
    actionFn = lambda state: mdp.getPossibleActions(state)
    qLearnOpts = {'gamma': opts.discount, 
                  'alpha': opts.learningRate, 
                  'epsilon': opts.epsilon,
                  'actionFn': actionFn}
    a = modularAgents.ReducedModularAgent(**qLearnOpts)
    a.setWeights(weights)

    if config.DISCRETE_Q:
      # way 1: using q tables
      a.setStateFilter(featureExtractors.getHumanDiscreteState(mdp))
      a.setQFuncs(modularQFuncs.getHumanWorldDiscreteFuncs())
    else:
      # way 2: using q functions
      a.setStateFilter(featureExtractors.getHumanContinuousState(mdp))
      a.setQFuncs(modularQFuncs.getHumanWorldQPotentialFuncs(discounters))
  elif opts.agent == 'random':
    import baselineAgents
    a = baselineAgents.RandomAgent()
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
    decisionCallback = lambda state : continuousWorldExperiment.getUserAction(state, mdp.getPossibleActions)
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
    continuousWorldPlot.plotHumanWorldQFuncs(a.values, opts.category)

  # hold window
  if not opts.quiet and 'vr' in opts.grid:
    handler = 'task_' + str(vrDomainId / 8 + 1) + '_room_' + str(vrDomainId + 1)
    win.postscript(file=handler+".eps")
    win.close()

if __name__ == '__main__':
  main()
