import gridworld
import numpy as np
from inverseBayesianRL import InverseBayesianRL
from policyIterationAgents import PolicyIterationAgent

def laplacePriorGen(sigma):
  return lambda r: 1.0 / (2 * sigma) * np.exp(- abs(r) / (2 * sigma))

def gaussianPriorGen(sigma):
  return lambda r: 1.0 / (np.sqrt(2 * np.pi) * sigma) * np.exp(- r ** 2 / (2 * sigma))

def gridWorldExperiments():
  #mdp = gridworld.getBookGrid()
  mdp = gridworld.getToyWalkAvoidGrid()
  
  opts = {'gamma': 0.8, 
          'actionFn': lambda state: mdp.getPossibleActions(state)}
  a = PolicyIterationAgent(mdp, **opts)
  a.learn()

  print [(state, a.getPolicy(state)) for state in mdp.getStates()]
  
  sol = InverseBayesianRL(mdp, laplacePriorGen(1))
  sol.setSamplesFromMdp(mdp, a)
  sol.solve()

def main():
  gridWorldExperiments()

if __name__ == '__main__':
  main()