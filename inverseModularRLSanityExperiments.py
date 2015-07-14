import modularQFuncs
import continuousWorldDomains
from humanWorld import HumanWorld
import modularAgents
from inverseModularRL import InverseModularRL
import featureExtractors

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
  agent.setParameters(gtX)

  starts = [0] * 2 + [0.5] * 2 + [0] * 2
  margin = 0.1
  bnds = ((0, 100), (-100, 0))\
       + tuple((0 + margin, 1 - margin) for _ in range(2))\
       + tuple((0, 1) for _ in range(2))
  # add constants for path module
  decorator = lambda x: x[0:2] + [0] + x[2:4] + [0] + x[4:6] + [0]
  
  sln = InverseModularRL(qFuncs, starts, bnds, decorator, solver="CMA-ES")
  sln.setSamplesFromMdp(mdp, agent)
  x = sln.solve()
  
  print "ground truth: ", gtX
  print "solved: ", x

  return x

if __name__ == '__main__':
  gtX = [4, -2, 0] + [.3, .3, 0] + [0, 0.2, 0]
  sanityCheck(gtX)