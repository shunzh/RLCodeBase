import continuousWorldDomains
import modularAgents
import modularQFuncs
from inverseModularRL import InverseModularRL
import util

def continuousWorldExperiment(x):
  """
    Can be called to run pre-specified agent and domain.
  """
  import continuousWorld as cw
  init = continuousWorldDomains.loadFromMat('miniRes25.mat', 0)
  m = cw.ContinuousWorld(init)

  actionFn = lambda state: m.getPossibleActions(state)
  qLearnOpts = {'gamma': 0.9,
                'alpha': 0.5,
                'epsilon': 0,
                'actionFn': actionFn}
  # modular agent
  a = modularAgents.ModularAgent(**qLearnOpts)

  qFuncs = modularQFuncs.getContinuousWorldFuncs(m)
  # set the weights and corresponding q-functions for its sub-mdps
  # note that the modular agent is able to determine the optimal policy based on these
  a.setQFuncs(qFuncs)

  sln = InverseModularRL(qFuncs)
  sln.setSamplesFromMdp(m, a)
  x = sln.solve()

  # check the consistency between the original optimal policy
  # and the policy predicted by the weights we guessed.
  aHat = modularAgents.ModularAgent(**qLearnOpts)
  aHat.setQFuncs(qFuncs)
  aHat.setParameters(x)

  # print for experiments
  print util.checkPolicyConsistency(m.getStates(), a, aHat)
  print util.getVectorDistance(a.getParameters(), x)

  return x, sln

def main():
  x = [1, 0, 0] + [.5, .5, .5]
  continuousWorldExperiment(x)

if __name__ == '__main__':
  main()