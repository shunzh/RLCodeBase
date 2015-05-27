import unittest
import qlearningAgents
import numpy as np
import gridworldExperiment
import policyIterationAgents

class Test(unittest.TestCase):
  def test_q_sidewalk(self):
    import gridworld
    mdp = gridworld.getSidewalkGrid()

    env = gridworld.GridworldEnvironment(mdp)
    actionFn = lambda state: mdp.getPossibleActions(state)
    gamma = 0.8
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
    
    self.sidewalk_verify(mdp, a, gamma)
    
  def test_pi_sidewalk(self):
    import gridworld
    mdp = gridworld.getSidewalkGrid()

    actionFn = lambda state: mdp.getPossibleActions(state)
    gamma = 0.8
    qLearnOpts = {'gamma': gamma, 
                  'actionFn': actionFn}
    # model based, need mdp
    a = policyIterationAgents.PolicyIterationAgent(mdp, **qLearnOpts)

    a.learn()
   
    self.sidewalk_verify(mdp, a, gamma)
 
  def sidewalk_verify(self, mdp, agent, gamma):
    for state in mdp.getStates():
      if state[0] < 4:
        policy = agent.getPolicy(state)
        value = agent.getValue(state)
        idealValue = np.power(gamma, 3 - state[0])
        self.assertAlmostEqual(value, idealValue,\
                               msg='Wrong value ' + str(value) + ' compared to ' + str(idealValue) + ' at state ' + str(state))
        self.assertTrue(policy == 'east',\
                        msg='Wrong policy ' + policy + ' at state ' + str(state))
    
if __name__ == '__main__':
  unittest.main()