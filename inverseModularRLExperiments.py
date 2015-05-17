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

def checkPolicyConsistency(states, a, b):
  """
    Check how many policies on the states are consistent with the optimal one.

    Args:
      states: the set of states that we want to compare the policies
      a, b: two agents that we want to compare their policies
    Return:
      Portion of consistent policies
  """
  consistentPolices = 0

  # Walk through each state
  for state in states:
    consistentPolices += int(a.getPolicy(state) == b.getPolicy(state))

  return 1.0 * consistentPolices / len(states)

def getWeightDistance(w1, w2):
  """
    Return:
      ||w1 - w2||_2
  """
  assert len(w1) == len(w2)

  return np.linalg.norm([w1[i] - w2[i] for i in range(len(w1))])

def continuousWorldExperiment():
  """
    Can be called to run pre-specified agent and domain.
  """
  import continuousWorld as cw
  init = continuousWorldDomains.loadFromMat('miniRes25.mat', 0)
  #init = cw.toyDomain()
  m = cw.ContinuousWorld(init)
  env = cw.ContinuousEnvironment(m)

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
  output = sln.solve()
  w = output.x.tolist()
  w = map(lambda _: round(_, 5), w) # avoid weird numerical problem

  print "Weight: ", w

  # check the consistency between the original optimal policy
  # and the policy predicted by the weights we guessed.
  aHat = modularAgents.ModularAgent(**qLearnOpts)
  aHat.setQFuncs(qFuncs)
  aHat.setWeights(w) # get the weights in the result

  # print for experiments
  print checkPolicyConsistency(m.getStates(), a, aHat)
  print getWeightDistance(a.getWeights(), w)

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

def evaluateAssumption(zippedData, qFuncs, w, d = None):
  """
  Given samples and weights, compare policies of human and our agents.
  Args:
    samples: list of (state, action)
    w: learned weight
    d: learned discounters
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
    agent.setWeights(w)
    if d != None: agent.setDiscounters(d)
    agent.setQFuncs(qFuncs)

    # go through samples
    angularDiff = 0
    posteriorProb = 0
    for datum, sample in zippedData:
      state, action = sample

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
  print rang, ": Started."
  qFuncs = modularQFuncs.getHumanWorldDiscreteFuncs()
  n = len(qFuncs)

  sln = InverseModularRL(qFuncs)
  parsedHumanData = humanInfoParser.parseHumanData(filenames, rang)
  samples = humanInfoParser.getHumanStatesActions(filenames, rang, parsedHumanData)
  sln.setSamples(samples, humanWorld.HumanWorld.actions.getActions())

  x = sln.solve()
  w = x[:n]
  # TODO test evaluation
  evaluation = evaluateAssumption(zip(parsedHumanData, samples), qFuncs, w)

  print rang, ": weights are", w
  print rang, ": evaluation", evaluation 

  printWeight(sln, 'objValuesTask' + str(rang[0] / len(rang) + 1) + '.png')
  print rang, ": weight heatmaps done."
  print rang, ": OK."

  return [w, evaluation] 

def humanWorldExperimentQPotential(filenames, rang, solving = True):
  """
  Args:
    rang: load mat with given rang of trials
  """
  print rang, ": Started."
  qFuncs = modularQFuncs.getHumanWorldQPotentialFuncs()
  n = len(qFuncs)

  sln = InverseModularRL(qFuncs, learnDiscounter=True)
  parsedHumanData = humanInfoParser.parseHumanData(filenames, rang)
  samples = humanInfoParser.getHumanStatesActions(filenames, rang, parsedHumanData)
  sln.setSamples(samples, humanWorld.HumanWorld.actions.getActions())

  if solving:
    x = sln.solve()
  else:
    # read from files in this case
    values = pickle.load(open('learnedValues/values.pkl'))
    x = values[rang[0] / 8]
  w = x[:n]
  d = x[n:]
  evaluation = evaluateAssumption(zip(parsedHumanData, samples), qFuncs, w, d)

  print w + d
  print evaluation 

  return [w + d, evaluation] 

def humanWorldExperimentQPotentialGridSearch(filenames, rang):
  """
  One way to solve the nonconvex objective function,
  grid search over all possible discounters, and optimize the weights for each
  """
  print rang, ": Started."
  qFuncs = modularQFuncs.getHumanWorldQPotentialFuncs()
  n = len(qFuncs)

  sln = InverseModularRL(qFuncs)
  samples = humanInfoParser.getHumanStatesActions(filenames, rang)
  sln.setSamples(samples, humanWorld.HumanWorld.actions.getActions())

  discounterRange = map(lambda x: x *.1,  range(1, 10))
  optD = []
  optFunValue = 0 # values are to be minimized, must be positive
  # FIXME overfit length of q
  for d1 in discounterRange:
    for d2 in discounterRange:
      for d3 in discounterRange:
        d = [d1, d2, d3]
        objValue, x = humanWorldExperimentQPotentialGridSearchHelper(sln, d)
       
        if sln.objValue > optFunValue:
          optW = x[:n]
          optD = d

  evaluation = evaluateAssumption(samples, qFuncs, optW, optD)

  print rang, ": weights are", optW
  print rang, ": discounters are", optD
  print rang, ": evaluation ", evaluation 

  return [optW + optD, evaluation] 

def humanWorldExperimentQPotentialGridSearchHelper(sln, d):
  sln.setDiscounters(d)
  x = sln.solve()
  
  return [sln.objValue, x]

def main():
  import config
  
  # set experiment here
  if config.DISCRETE_Q:
    experiment = humanWorldExperimentDiscrete
  else:
    #experiment = humanWorldExperimentQPotentialGridSearch
    experiment = humanWorldExperimentQPotential

  from multiprocessing import Pool
  # change the number of processors used here.
  # use 1 for sequential execution.
  pool = Pool(processes=1)

  subjFiles = ["subj" + str(num) + ".parsed.mat" for num in xrange(25, 29)]
  taskRanges = [range(0, 8), range(8, 16), range(16, 24), range(24, 32)]

  if len(sys.argv) > 1:
    taskId = int(sys.argv[1])
    values, evaluations = experiment(subjFiles, taskRanges[taskId])

    util.saveToFile('values' + str(taskId) + '.pkl', values)
    util.saveToFile('evaluation' + str(taskId) + '.pkl', evaluations)
  else:
    results = [pool.apply_async(experiment, args=(subjFiles, ids)) for ids in taskRanges]
    values = [r.get()[0] for r in results]
    evaluations = [r.get()[1] for r in results]

    util.saveToFile('values.pkl', values)
    util.saveToFile('evaluation.pkl', evaluations)

if __name__ == '__main__':
  main()