import numpy as np
import sys

from inverseModularRL import InverseModularRL
import modularAgents
import modularQFuncs
import humanInfoParser
import continuousWorldDomains
import baselineAgents
import humanWorld
import pickle
import util
import config
import warnings

def continuousWorldExperiment():
  """
    Can be called to run pre-specified agent and domain.
  """
  import continuousWorld as cw
  init = continuousWorldDomains.loadFromMat('miniRes25.mat', 0)
  #init = cw.toyDomain()
  m = cw.ContinuousWorld(init)

  actionFn = lambda state: m.getPossibleActions(state)
  qLearnOpts = {'gamma': 0.9,
                'alpha': 0.5,
                'epsilon': 0,
                'actionFn': actionFn}
  # modular agent
  a = modularAgents.ModularAgent(**qLearnOpts)

  if len(sys.argv) > 1:
    # user wants to set weights themselves
    w = map(float, sys.argv[1:])
    a.setWeights(w)

  qFuncs = modularQFuncs.getContinuousWorldFuncs(m)
  # set the weights and corresponding q-functions for its sub-mdps
  # note that the modular agent is able to determine the optimal policy based on these
  a.setQFuncs(qFuncs)

  sln = InverseModularRL(qFuncs)
  sln.setSamplesFromMdp(m, a)
  w = sln.solve()
  w = map(lambda _: round(_, 5), w) # avoid weird numerical problem

  # check the consistency between the original optimal policy
  # and the policy predicted by the weights we guessed.
  aHat = modularAgents.ModularAgent(**qLearnOpts)
  aHat.setQFuncs(qFuncs)
  aHat.setWeights(w) # get the weights in the result

  # print for experiments
  print util.checkPolicyConsistency(m.getStates(), a, aHat)
  print util.getVectorDistance(a.getWeights(), w)

  return w, sln

def printWeight(sln, filename, discounters = []):
  import matplotlib.pyplot as plt

  stepSize = 2
  data = []
  for i in range(0, 11, stepSize):
    row = []
    for j in range(0, 11 - i, stepSize):
      k = 10 - i - j
      row.append(-sln.obj([0.1 * i, 0.1 * j, 0.1 * k] + discounters))
    for j in range(11 - i, 11, stepSize):
      row.append(0) # will be masked
    data.append(row)

  mask = np.tri(10 / stepSize + 1, k=-1)
  mask = np.fliplr(mask)
  data = np.ma.array(data, mask=mask)

  plt.imshow(data, interpolation='none')
  plt.xticks(range(6), np.arange(0,1.1,0.1 * stepSize))
  plt.yticks(range(6), np.arange(0,1.1,0.1 * stepSize))
  plt.xlabel('Obstacle Module Weight');
  plt.ylabel('Target Module Weight');

  plt.jet()
  plt.colorbar()

  plt.savefig(filename)
  
  plt.close()

def printDiscounter(sln, filename, weights = []):
  import matplotlib.pyplot as plt
  
  stepSize = 2
  data = []
  for i in range(0, 9, stepSize):
    row = []
    for j in range(0, 9, stepSize):
      row.append(-sln.obj(weights + [0.1 * i, 0.1 * j, 0.8]))
    data.append(row)

  data = np.ma.array(data)
  plt.imshow(data, interpolation='none')
  plt.xticks(range(5), np.arange(0,0.9,0.1 * stepSize))
  plt.yticks(range(5), np.arange(0,0.9,0.1 * stepSize))
  plt.xlabel('Obstacle Module Discounter');
  plt.ylabel('Target Module Discounter');

  plt.jet()
  plt.colorbar()

  plt.savefig(filename)
  
  plt.close()

def evaluateAssumption(zippedData, qFuncs, x):
  """
  Given samples and weights, compare policies of human and our agents.
  Args:
    zippedData: raw data and parsed data
    qFuncs
    x: learned parameters
  Return:
    A dictionary {agent: {criteria: value}}
  """
  # define agent
  actionFn = lambda state: humanWorld.HumanWorld.actions.getActions()
  qLearnOpts = {'gamma': 0.9,
                'alpha': 0.5,
                'epsilon': 0,
                'actionFn': actionFn}

  def evaluate(agentType):
    """
    Arg:
      class of agent.
    Return:
      dict of evaluation results this agent.
      {key: value}
    """
    agent = agentType(**qLearnOpts)
    agent.setQFuncs(qFuncs)
    if hasattr(agent, 'setParameters'):
      agent.setParameters(x)

    # go through samples
    angularDiff = 0
    posteriorProb = 0
    for datum, sample in zippedData:
      state, action = sample

      # difference in angle
      angularDiff += abs(humanWorld.HumanWorld.actions.getExpectedDistAngle(agent.getPolicy(state))[1] - datum['moveAngle'])
      # add log of the probability of choosing such action by the model
      posteriorProb += np.log(agent.getPolicyProbability(state, action))
    angularDiff = (1.0 * angularDiff / len(zippedData)) / np.pi * 180
    return {'angularDifference': angularDiff,\
            'likelihood': posteriorProb}
  
  candidates = [modularAgents.ModularAgent, baselineAgents.RandomAgent, baselineAgents.ReflexAgent]
  return {candidate.__name__: evaluate(candidate) for candidate in candidates}

def humanWorldExperimentDiscrete(filenames, rang):
  """
  Args:
    rang: load mat with given rang of trials
  """
  qFuncs = modularQFuncs.getHumanWorldDiscreteFuncs()
  n = len(qFuncs)

  sln = InverseModularRL(qFuncs)
  parsedHumanData = humanInfoParser.parseHumanData(filenames, rang)
  samples = humanInfoParser.getHumanStatesActions(filenames, rang, parsedHumanData)
  sln.setSamples(samples, humanWorld.HumanWorld.actions.getActions())

  x = sln.solve()
  # FIXME need both raw data and parsed data, so zip them together
  evaluation = evaluateAssumption(zip(parsedHumanData, samples), qFuncs, x)

  printWeight(sln, 'objValuesTask' + str(rang[0] / len(rang) + 1) + '.png')

  return [x, evaluation] 

def humanWorldExperimentQPotential(filenames, rang, solving = True):
  """
  Args:
    rang: load mat with given rang of trials
  """
  qFuncs = modularQFuncs.getHumanWorldQPotentialFuncs()
  n = len(qFuncs)

  starts = [0] * n + [0.5] * n + [0] * n
  margin = 0.1
  bnds = ((0, 1000), (-1000, 0), (0, 1000))\
       + tuple((0 + margin, 1 - margin) for _ in range(n))\
       + tuple((0, 0.001) for _ in range(n))

  sln = InverseModularRL(qFuncs, starts, bnds, solver="CMA-ES")
  parsedHumanData = humanInfoParser.parseHumanData(filenames, rang)
  samples = humanInfoParser.getHumanStatesActions(filenames, rang, parsedHumanData)
  sln.setSamples(samples, humanWorld.HumanWorld.actions.getActions())

  if solving:
    x = sln.solve()
  else:
    # read from files if only output the learned results
    values = pickle.load(open('learnedValues/values.pkl'))
    x = values[rang[0] / 8]
  
  evaluation = evaluateAssumption(zip(parsedHumanData, samples), qFuncs, x)

  return [x, evaluation] 

def main():
  # set experiment here
  experiment = humanWorldExperimentQPotential

  # change the number of processors used here.
  # use 1 for sequential execution.

  subjFiles = ["subj" + str(num) + ".parsed.mat" for num in config.HUMAN_SUBJECTS]
  taskRanges = [range(0, 8), range(8, 16), range(16, 24), range(24, 32)]

  """
  # run trials separately
  trialId = int(sys.argv[1])

  subjIdx = trialId / taskNum
  taskIdx = trialId % taskNum

  results = experiment([subjFiles[subjIdx]], [taskIdx])
  print results[0]
  """
  
  # run tasks separately
  taskId = int(sys.argv[1])
  value, evaluation = experiment(subjFiles, taskRanges[taskId])

  util.saveToFile('values' + str(taskId) + '.pkl', value)
  util.saveToFile('evaluation' + str(taskId) + '.pkl', evaluation)

if __name__ == '__main__':
  main()
