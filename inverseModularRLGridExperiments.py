import sys
import modularQFuncs
import modularAgents
import config
from inverseModularRL import InverseModularRL
import featureExtractors

def experiment(gtX):
  # set budget (number of samples used) as the first argument
  if len(sys.argv) > 1:
    budgetId = int(sys.argv[1]) / 10
    budget = config.BUDGET_SIZES[budgetId]
  else:
    budget = None

  import gridworldMaps
  mdp = gridworldMaps.getRuohanGrid(0)
  qFuncs = modularQFuncs.getGridQPotentialFuncs(mdp)

  actionFn = lambda state: mdp.getPossibleActions(state)
  qLearnOpts = {'gamma': 0.9,
                'alpha': 0.5,
                'epsilon': 0,
                'actionFn': actionFn}
  # modular agent
  a = modularAgents.ModularAgent(**qLearnOpts)
  a.setQFuncs(qFuncs)
  a.setParameters(gtX)

  starts = [0] * 2 + [0.5] * 2
  margin = 0.01
  bnds = ((0, 1), (0, 1))\
       + tuple((0 + margin, 1 - margin) for _ in range(2))

  sln = InverseModularRL(qFuncs, starts, bnds, solver="CMA-ES")
  sln.setSamplesFromMdp(mdp, a, budget)
  x = sln.solve()
  print x

if __name__ == '__main__':
  x = [1, 1, .9, .1]
  experiment(x)