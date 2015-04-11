import unittest
import qlearningAgents
import numpy as np
import gridworldExperiment

class Test(unittest.TestCase):
  def test_sidewalk(self):
    import gridworld
    mdp = gridworld.getSidewalkGrid()
    mdp.setNoise(0)

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

    for episode in xrange(500):
      gridworldExperiment.runEpisode(a, env, gamma, a.getAction, displayCallback, messageCallback, pauseCallback, episode)
    
    for state in mdp.getStates():
      if state[0] < 4:
        policy = a.getPolicy(state)
        value = a.getValue(state)
        idealValue = np.power(gamma, 3 - state[0])
        self.assertAlmostEqual(value, idealValue, msg='Wrong value ' + str(value) + ' compared to ' + str(idealValue) + ' at state ' + str(state))
        self.assertTrue(policy == 'east', msg='Wrong policy ' + policy + ' at state ' + str(state))
    
if __name__ == '__main__':
  unittest.main()