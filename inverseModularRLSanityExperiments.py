import modularQFuncs
import continuousWorldDomains
from humanWorld import HumanWorld
import modularAgents
from inverseModularRL import InverseModularRL
import featureExtractors
import sys
import numpy as np

def plotContinuousDomainValues(mdp, agent, mapper, filename):
  import matplotlib.pyplot as plt

  stepSize = 0.1
  xBoundary = mdp.xBoundary
  yBoundary = mdp.yBoundary
  
  data = []
  x = xBoundary[0]
  while x < xBoundary[1]:
    row = []
    y = yBoundary[0]
    while y < yBoundary[1]:
      # mapper maps (s, a) to (s, a). only need s here
      state = ((x, y), 0)
      row.append(agent.getValue(state))
      y += stepSize
    data.append(row)
    x += stepSize

  plt.imshow(data, interpolation='none')
  plt.xticks(range(xBoundary[1] - xBoundary[0]), np.arange(xBoundary[0],xBoundary[1],stepSize))
  plt.yticks(range(yBoundary[1] - yBoundary[0]), np.arange(yBoundary[0],yBoundary[1],stepSize))

  plt.jet()
  plt.colorbar()

  plt.savefig(filename)
  
  plt.close()

def sanityCheck(gtX, id):
  qFuncs = modularQFuncs.getHumanWorldQPotentialFuncs()

  init = continuousWorldDomains.simpleMixDomain()
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

  starts = [0] * 2 + [0.5] * 2 + [0] * 2
  margin = 0.1
  bnds = ((0, 1), (-1, 0))\
       + tuple((0 + margin, 1 - margin) for _ in range(2))\
       + tuple((0, 1) for _ in range(2))
  # add constants for path module
  decorator = lambda x: x[0:2] + [0] + x[2:4] + [0] + x[4:6] + [0]
  
  agent.setParameters(decorator(gtX))

  sln = InverseModularRL(qFuncs, starts, bnds, decorator, solver="CMA-ES")
  sln.setSamplesFromMdp(mdp, agent)
  x = sln.solve()
  
  plotContinuousDomainValues(mdp, agent, mapper, "continuous_world_values" + str(id) + ".png")
  
  print "results:"
  print gtX
  print x

  return x

if __name__ == '__main__':
  taskId = int(sys.argv[1])
  
  gtXs = [[1, -1] + [.1, .9] + [0, 0],\
          [1, -1] + [.9, .1] + [0, 0],\
          [1, -1] + [.1, .9] + [0.8, 0],\
          [1, -1] + [.1, .9] + [0, 0.8]]
  sanityCheck(gtXs[taskId], taskId)