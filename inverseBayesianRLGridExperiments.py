import numpy as np
from inverseBayesianRL import InverseBayesianRL
from policyIterationAgents import PolicyIterationAgent
import gridworldMaps
import sys
import config

# two popular reward priors
def laplacePriorGen(sigma):
  return lambda r: 1.0 / (2 * sigma) * np.exp(- abs(r) / (2 * sigma))

def gaussianPriorGen(sigma):
  return lambda r: 1.0 / (np.sqrt(2 * np.pi) * sigma) * np.exp(- r ** 2 / (2 * sigma))

def experiment(mdp):
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
  return sol.solve()

def inferWeightsFromReward(mdp, rewards):
  states = mdp.getStates()
  weights = [] 
  for r, c in mdp.spec:
    modularReward = 0.0
    for stateIdx in xrange(len(states)):
      state = states[stateIdx]
      x, y = state
      cell = mdp.grid[x][y] 
      if cell == r:
        modularReward += abs(rewards[stateIdx])
    modularReward /= c
    weights.append(modularReward)
  
  return map(lambda _: _ / sum(weights), weights)

def main():
  #mdp = gridworldMaps.getToyWalkAvoidGrid()
  #mdp = gridworldMaps.getBookGrid()
  mdp = gridworldMaps.getRuohanGrid()

  rewards = experiment(mdp)
  print inferWeightsFromReward(mdp, rewards)

if __name__ == '__main__':
  main()