"""
Inverse MRL with gridsearch on discounters
For running on Condor.
"""
import sys
import inverseModularRLExperiments
import modularQFuncs
import inverseModularRL
import humanWorld
import humanInfoParser

if __name__ == '__main__':
  """
  Takes input arguments as discounters specification.
  """
  procId = int(sys.argv[1])
  
  d3 = procId % 10
  d2 = (procId / 10) % 10
  d1 = (procId / 100) % 10
  taskId = procId / 1000
  
  if d1 == 0 or d2 == 0 or d3 == 0 or taskId >= 4:
    # drop insensible discounters
    exit()
    
  d = [d1 * .1, d2 * .1, d3 * .1]

  subjFiles = ["subj" + str(num) + ".parsed.mat" for num in xrange(25, 29)]
  taskRanges = [range(0, 8), range(8, 16), range(16, 24), range(24, 32)]
  qFuncs = modularQFuncs.getHumanWorldQPotentialFuncs()

  sln = inverseModularRL.InverseModularRL(qFuncs)
  samples = humanInfoParser.getHumanStatesActions(subjFiles, taskRanges[taskId])
  sln.setSamples(samples, humanWorld.HumanWorld.actions.getActions())
  experiment = inverseModularRLExperiments.humanWorldExperimentQPotentialGridSearchHelper

  print experiment(sln, d)