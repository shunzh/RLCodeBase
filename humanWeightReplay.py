import inverseModularRL
import modularAgents
import continuousWorld, humanWorld
import util

import numpy as np

def main():
  [w, sln] = inverseModularRL.humanWorldExperiment("subj25.parsed.mat", range(25, 31))

  qLearnOpts = {'gamma': 0.9,
                'alpha': 0.5,
                'epsilon': 0}
  qFuncs = modularAgents.getHumanWorldContinuousFuncs()
  aHat = modularAgents.ModularAgent(**qLearnOpts)
  aHat.setQFuncs(qFuncs)
  aHat.setWeights(w) # get the weights in the result

  # plot domain and policy
  init = continuousWorld.loadFromMat('miniRes25.mat', 24)
  mdp = humanWorld.HumanWorld(init)

  win = continuousWorld.drawDomain(mdp)
  # parse human positions and actions
  humanSamples = parseHumanActions("subj25.parsed.mat", 25)
  # use IRL code to get features
  featureSamples = inverseModularRL.getSamplesFromMat("subj25.parsed.mat", [25])

  assert len(humanSamples) == len(featureSamples)

  for i in range(len(humanSamples)):
    print humanSamples[i]['action'], featureSamples[i][1]

  win.getMouse()
  win.close()

def parseHumanActions(filename, domainId):
  """
  Parse human behavivors from mat, which includes at which point, take which action.

  Args:
    filename: mat file to read
    domainId: room number

  Return:
    a list of dicts, which includes x, y, orient and action.
  """
  mat = util.loadmat(filename)

  samples = []

  agentXs = mat['pRes'][domainId].agentX
  agentZs= mat['pRes'][domainId].agentZ
  agentAngles = mat['pRes'][domainId].agentAngle / 180 * np.pi
  actions = mat['pRes'][domainId].action

  for i in range(len(agentXs)):
    samples.append({'x': agentXs[i], 'y': agentZs[i], 'orient': agentAngles[i], 'action': actions[i]})

  return samples

if __name__ == '__main__':
  main()
