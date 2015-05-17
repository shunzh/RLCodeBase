import gridworld
import qlearningAgents
import gridworldExperiment
import numpy as np
from inverseBayesianRL import InverseBayesianRL

def laplacePriorGen(sigma):
  return lambda r: 1.0 / (2 * sigma) * np.exp(abs(r) / (2 * sigma))

def gaussianPriorGen(sigma):
  return lambda r: 1.0 / (np.sqrt(2 * np.pi) * sigma) * np.exp(- r ** 2 / (2 * sigma))

def gridWorldExperiments():
  mdp = gridworld.getToyWalkAvoidGrid()
  
  # learn the policy first
  env = gridworld.GridworldEnvironment(mdp)
  actionFn = lambda state: mdp.getPossibleActions(state)
  gamma = 0.9
  qLearnOpts = {'gamma': gamma, 
                'alpha': 0.5, 
                'epsilon': 0.5,
                'actionFn': actionFn}
  a = qlearningAgents.QLearningAgent(**qLearnOpts)

  displayCallback = lambda state: None
  messageCallback = lambda state: None
  pauseCallback = lambda: None
  for episode in xrange(100):
    gridworldExperiment.runEpisode(a, env, gamma, a.getAction, displayCallback, messageCallback, pauseCallback, episode)
  
  sol = InverseBayesianRL(mdp, laplacePriorGen(1))
  sol.setSamplesFromMdp(mdp, a)
  sol.solve()

def main():
  gridWorldExperiments()

if __name__ == '__main__':
  main()