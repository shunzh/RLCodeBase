import numpy as np
import sys

from inverseModularRL import InverseModularRL
import modularAgents
import modularQFuncs
import humanInfoParser
import continuousWorldDomains

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

def discretize(samples):
  """
  Map samples to bins
  """
  import featureExtractors
  ret = []
  for sample in samples:
    s = map(lambda (d, a): featureExtractors.mapStateToBin((d, a)), sample[0])
    a = sample[1]
    ret.append((s, a))
  return ret

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
  for i in range(0, 11, stepSize):
    row = []
    for j in range(0, 11, stepSize):
      row.append(-sln.obj(weights + [0.1 * i, 0.1 * j, 0.8]))
    data.append(row)

  data = np.ma.array(data)
  plt.imshow(data, interpolation='none')
  plt.xticks(range(6), np.arange(0,1.1,0.1 * stepSize))
  plt.yticks(range(6), np.arange(0,1.1,0.1 * stepSize))
  plt.xlabel('Obstacle Module Discounter');
  plt.ylabel('Target Module Discounter');

  plt.jet()
  plt.colorbar()

  plt.savefig(filename)
  
  plt.close()

def policyCompare(samples, qFuncs, w):
  """
  Given samples and weights, compare policies of human and our agent.
  #FIXME discounter not provided!
  Args:
    samples: list of (state, action)
    w: weight learned
  Return:
    Proportion of agreed policies
  """
  # define agent
  import humanWorld
  actionFn = lambda state: humanWorld.HumanWorld.actions
  qLearnOpts = {'gamma': 0.9,
                'alpha': 0.5,
                'epsilon': 0,
                'actionFn': actionFn}
  a = modularAgents.ModularAgent(**qLearnOpts)
  a.setWeights(w)
  a.setQFuncs(qFuncs)

  # go through samples
  agreedPolicies = 0
  for state, action in samples:
    agreedPolicies += a.getPolicy(state) == action 

  return 1.0 * agreedPolicies / len(samples)

def humanWorldExperimentDiscrete(filenames, rang):
  """
  Args:
    rang: load mat with given rang of trials
  """
  print rang, ": Started."
  qFuncs = modularQFuncs.getHumanWorldDiscreteFuncs()
  n = len(qFuncs)

  sln = InverseModularRL(qFuncs)
  samples = humanInfoParser.getHumanStatesActions(filenames, rang)
  samples = discretize(samples)
  sln.setSamples(samples)

  x = sln.solve()
  w = x[:n]
  d = x[n:]
  agreedPoliciesRatio = policyCompare(samples, qFuncs, w)

  print rang, ": weights are", w
  print rang, ": proportion of agreed policies ", agreedPoliciesRatio 

  # debug weight disabled. computational expensive?
  printWeight(sln, 'objValuesTask' + str(rang[0] / len(rang) + 1) + '.png')
  print rang, ": weight heatmaps done."
  print rang, ": OK."

  return [w + d, agreedPoliciesRatio] 

def humanWorldExperimentQPotential(filenames, rang):
  """
  Args:
    rang: load mat with given rang of trials
  """
  print rang, ": Started."
  qFuncs = modularQFuncs.getHumanWorldQPotentialFuncs()
  n = len(qFuncs)

  sln = InverseModularRL(qFuncs)
  samples = humanInfoParser.getHumanStatesActions(filenames, rang)
  sln.setSamples(samples)

  x = sln.solve()
  w = x[:n]
  d = x[n:]
  # FIXME
  agreedPoliciesRatio = policyCompare(samples, qFuncs, w)

  print rang, ": weights are", w
  print rang, ": discounters are", d
  print rang, ": proportion of agreed policies ", agreedPoliciesRatio 

  # debug weight disabled. computational expensive?
  printWeight(sln, 'objValuesTask' + str(rang[0] / len(rang) + 1) + '.png', d)
  print rang, ": weight heatmaps done."
  if sln.learnDiscounter:
    printDiscounter(sln, 'discounterTask' + str(rang[0] / len(rang) + 1) + '.png', w)
    print rang, ": discounter heatmaps done."
  print rang, ": OK."

  return [w + d, agreedPoliciesRatio] 

if __name__ == '__main__':
  # set experiment here
  #experiment = humanWorldExperimentDiscrete
  experiment = humanWorldExperimentQPotential
  
  from multiprocessing import Pool
  # change the number of processors used here.
  # use 1 for sequential execution.
  pool = Pool(processes=1)

  subjFiles = ["subj" + str(num) + ".parsed.mat" for num in xrange(25, 29)]
  taskRanges = [range(0, 8), range(8, 16), range(16, 24), range(24, 31)]
  
  results = [pool.apply_async(experiment, args=(subjFiles, ids)) for ids in taskRanges]

  import pickle
  weights = [r.get()[0] for r in results]
  agreedPoliciesRatios = [r.get()[1] for r in results]

  output = open('values.pkl', 'wb')
  pickle.dump(weights, output)
  output.close()

  output = open('agreedPolicies.pkl', 'wb')
  pickle.dump(agreedPoliciesRatios, output)
  output.close()
