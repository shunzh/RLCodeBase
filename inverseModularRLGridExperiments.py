import sys
import modularQFuncs
import modularAgents
import config
from inverseModularRL import InverseModularRL
import util

def main():
  if len(sys.argv) > 1:
    budgetId = int(sys.argv[1]) / 10
    budget = config.BUDGET_SIZES[budgetId]
  else:
    budget = None

  import gridworldMaps
  mdp = gridworldMaps.getRuohanGrid(0)
  qFuncs = modularQFuncs.getObsAvoidFuncs(mdp)

  trueW = [abs(w) for w, count in mdp.spec]
  trueW = map(lambda _: 1.0 * _ / sum(trueW), trueW)

  actionFn = lambda state: mdp.getPossibleActions(state)
  qLearnOpts = {'gamma': 0.9,
                'alpha': 0.5,
                'epsilon': 0,
                'actionFn': actionFn}
  # modular agent
  a = modularAgents.ModularAgent(**qLearnOpts)
  a.setQFuncs(qFuncs)
  a.setWeights(trueW)
  a.setDiscounters([.8] * len(qFuncs))

  sln = InverseModularRL(qFuncs, learnDiscounter=False, solver="BFGS")
  sln.setSamplesFromMdp(mdp, a, budget)
  w = sln.solve()

  print util.getMSE(w, trueW)

if __name__ == '__main__':
  main()