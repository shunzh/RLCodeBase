import modularQFuncs
import continuousWorldDomains
from humanWorld import HumanWorld
import modularAgents
from inverseModularRL import InverseModularRL
import featureExtractors

def sanityCheck(gtX):
  qFuncs = modularQFuncs.getHumanWorldQPotentialFuncs()
  n = len(qFuncs)

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

  starts = [0] * n + [0.5] * n + [0] * n
  margin = 0.1
  bnds = ((0, 1000), (-1000, 0), (0, 1000))\
       + tuple((0 + margin, 1 - margin) for _ in range(n))\
       + tuple((0, 1) for _ in range(n))
  
  sln = InverseModularRL(qFuncs, starts, bnds, solver="CMA-ES")
  sln.setSamplesFromMdp(mdp, agent)
  x = sln.solve()
  
  print "ground truth: ", gtX
  print "solved: ", x

  return x

if __name__ == '__main__':
  gtX = [1, -1, 0] + [.5, .5, 0] + [0, 0.5, 0]
  sanityCheck(gtX)