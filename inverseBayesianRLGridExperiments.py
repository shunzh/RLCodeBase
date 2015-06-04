import numpy as np
from inverseBayesianRL import InverseBayesianRL
from policyIterationAgents import PolicyIterationAgent
import gridworldMaps
import sys
import config
import util

# two popular reward priors
def laplacePriorGen(sigma):
  return lambda r: 1.0 / (2 * sigma) * np.exp(- abs(r) / (2 * sigma))

def gaussianPriorGen(sigma):
  return lambda r: 1.0 / (np.sqrt(2 * np.pi) * sigma) * np.exp(- r ** 2 / (2 * sigma))

def experiment(mdp):
  """
  Run bayes irl experiment.
  
  Return:
    r: reward
    consistency
  """
  if len(sys.argv) > 1:
    budgetId = int(sys.argv[1]) / 10
    budget = config.BUDGET_SIZES[budgetId]
  else:
    budget = None

  opts = {'gamma': 0.8, 
          'actionFn': lambda state: mdp.getPossibleActions(state)}
  a = PolicyIterationAgent(mdp, **opts)
  a.learn()

  sol = InverseBayesianRL(mdp, laplacePriorGen(1))
  sol.setSamplesFromMdp(mdp, a, budget)
  r = sol.solve()
  
  for idx, s in enumerate(mdp.getStates()):
    mdp.setReward(s, r[idx])
  aEst = PolicyIterationAgent(mdp, **opts)
  aEst.learn()
  
  return [r, util.checkPolicyConsistency(mdp.getStates(), a, aEst)]

def main():
  #mdpName = gridworldMaps.getToyWalkAvoidGrid
  mdpName = lambda: gridworldMaps.getRuohanGrid(seed=0)

  rewards, consistency = experiment(mdpName())

  print consistency

if __name__ == '__main__':
  main()