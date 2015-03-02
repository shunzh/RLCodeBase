"""
This is the test for modular IRL.
Try some simple domains, and human world experiments on both discrete and continuous Q functions.
"""
import unittest
import numpy as np
from inverseModularRL import InverseModularRL
import modularQFuncs

class Test(unittest.TestCase):
  def test_corridor(self):
    """
    Two objectives - going to left or going to right in a corridor.
    """
    actions = ['L', 'R']
    samples = [[(p, actions[0]) for p in range(2)], [(p, actions[1]) for p in range(2)]]
    qFuncs = [lambda s, a, d = None: 1 if a == actions[0] else 0,\
              lambda s, a, d = None: 1 if a == actions[1] else 0]
    resultConstraints = [lambda w: w[0] >= 0.99 and w[1] <= 0.01,\
                        lambda w: w[1] >= 0.99 and w[0] <= 0.01]
    self.checkResult(samples, actions, qFuncs, resultConstraints)
    
  def test_three_way(self):
    """
    One state and three actions
    """
    actions = range(3)

    samples = [[(0, action)] for action in actions]
    qFuncs = [lambda s, a, d = None: 1 if a == actions[0] else 0,\
              lambda s, a, d = None: 1 if a == actions[1] else 0,\
              lambda s, a, d = None: 1 if a == actions[2] else 0]
    resultConstraints = [lambda w: w[0] >= 0.99,\
                         lambda w: w[1] >= 0.99,\
                         lambda w: w[2] >= 0.99]
    self.checkResult(samples, actions, qFuncs, resultConstraints)
   
  def test_human_world_QPotential_two_modules(self):
    resultConstraints = [lambda w: w[0] >= 0.9,\
                         lambda w: w[1] >= 0.9,\
                         lambda w: True]
    self.human_world_continuous(modularQFuncs.getHumanWorldQPotentialFuncs()[:2], resultConstraints)

  def test_human_world_discrete_two_modules(self):
    resultConstraints = [lambda w: w[0] >= 0.9,\
                         lambda w: w[1] >= 0.9,\
                         lambda w: True]
    self.human_world_discrete(modularQFuncs.getHumanWorldDiscreteFuncs()[:2], resultConstraints)

  def test_human_world_QPotential(self):
    resultConstraints = [lambda w: w[0] >= 0.9,\
                         lambda w: w[1] >= 0.9,\
                         lambda w: w[2] >= 0.9]
    self.human_world_continuous(modularQFuncs.getHumanWorldQPotentialFuncs(), resultConstraints)

  def test_human_world_discrete(self):
    resultConstraints = [lambda w: w[0] >= 0.9,\
                         lambda w: w[1] >= 0.9,\
                         lambda w: w[2] >= 0.9]
    self.human_world_discrete(modularQFuncs.getHumanWorldDiscreteFuncs(), resultConstraints)

  def human_world_discrete(self, qFuncs, resultConstraints):
    """
    make some human data, which have clear intentions (to target, or avoid obstacle)
    """
    actions = ['L', 'G', 'R']
    # samples: targest, obstacles, path
    state = ((4, 2), None, (4, 2), None, (4, 0))
    # samples are: going to target, avoid obstacle, and going to path segment
    samples = [[(state, 'R')], [(state, 'L')], [(state, 'G')]]
    self.checkResult(samples, actions, qFuncs, resultConstraints)

  def human_world_continuous(self, qFuncs, resultConstraints):
    """
    make some human data, which have clear intentions (to target, or avoid obstacle)
    """
    actions = ['L', 'G', 'R']
    # samples: targest, obstacles, path
    biasAngle = 60.0 / 180 * np.pi
    state = ((0.6, biasAngle), None, (0.6, biasAngle), None, (0.6, 0))
    # samples are: going to target, avoid obstacle, and going to path segment
    samples = [[(state, 'R')], [(state, 'L')], [(state, 'G')]]
    self.checkResult(samples, actions, qFuncs, resultConstraints)

  def checkResult(self, samples, actions, qFuncs, resultConstraints):
    for expIdx in range(len(samples)):
      sln = InverseModularRL(qFuncs)
      sln.getSamples = lambda : samples[expIdx]
      sln.getActions = lambda s: actions
      output = sln.solve()
      w = output.x.tolist()
      
      self.assertTrue(resultConstraints[expIdx](w), msg="Exp #" + str(expIdx) + " weights: " + str(w))

if __name__ == '__main__':
  unittest.main()