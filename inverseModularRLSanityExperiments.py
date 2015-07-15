import modularQFuncs
import continuousWorldDomains
from humanWorld import HumanWorld
import modularAgents
from inverseModularRL import InverseModularRL
import featureExtractors
import sys

def sanityCheck(gtX):
  qFuncs = modularQFuncs.getHumanWorldQPotentialFuncs()

  init = continuousWorldDomains.loadFromMat('miniRes25.mat', 0)
  mdp = HumanWorld(init)
  actionFn = lambda state: mdp.getPossibleActions(state)
  qLearnOpts = {'gamma': 0.9,
                'alpha': 0.5,
                'epsilon': 0,
                'actionFn': actionFn}
  agent = modularAgents.ModularAgent(**qLearnOpts)
  mapper = featureExtractors.getHumanContinuousMapper(mdp)
  agent.setQFuncs(qFuncs)
  agent.setMapper(mapper) # absolute position to relative position to objects 

  starts = [0] * 2 + [0.5] * 2
  margin = 0.1
  bnds = ((0, 100), (-100, 0))\
       + tuple((0 + margin, 1 - margin) for _ in range(2))\
  # add constants for path module
  decorator = lambda x: x[0:2] + [0] + x[2:4] + [0] + [0] * 3
  
  agent.setParameters(decorator(gtX))

  sln = InverseModularRL(qFuncs, starts, bnds, decorator, solver="CMA-ES")
  sln.setSamplesFromMdp(mdp, agent)
  x = sln.solve()
  
  print "results:"
  print gtX
  print x

  return x

if __name__ == '__main__':
  taskId = int(sys.argv[1])
  
  gtXs = [[0.5, -0.5] + [.5, .5],\
          [0.5, -0.5] + [.5, .9],\
          [0.9, -0.1] + [.5, .5],\
          [0.1, -0.9] + [.5, .5]]
  sanityCheck(gtXs[taskId])