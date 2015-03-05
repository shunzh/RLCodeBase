"""
This is the test for modular IRL.
Try some simple domains, and human world experiments on both discrete and continuous Q functions.
"""
import unittest
import numpy as np
from inverseModularRL import InverseModularRL
import modularQFuncs
import inverseModularRLExperiments

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
    One state and three actions. One module for taking each action.
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
   
  def test_human_world_object_aside(self):
    actions = ['L', 'G', 'R']

    # samples: targest, obstacles, path
    step = 0.6
    biasAngle = 60.0 / 180 * np.pi
    state = ((step, biasAngle), None, (step, biasAngle), None, (step, 0))
    # samples are: going to target, avoid obstacle, and going to path segment
    samples = [[(state, 'R')], [(state, 'L')], [(state, 'G')]]
    qFuncs = modularQFuncs.getHumanWorldQPotentialFuncs()

    resultConstraints = [lambda w: w[0] >= 0.9,\
                         lambda w: w[1] >= 0.9,\
                         lambda w: w[2] >= 0.9]
    self.checkResult(samples, actions, qFuncs, resultConstraints)

  def test_human_world_object_aside_discrete(self):
    # samples: targest, obstacles, path
    actions = ['L', 'G', 'R']
    state = ((4, 2), None, (4, 2), None, (4, 0))
    # samples are: going to target, avoid obstacle, and going to path segment
    samples = [[(state, 'R')], [(state, 'L')], [(state, 'G')]]
    qFuncs = modularQFuncs.getHumanWorldDiscreteFuncs()

    resultConstraints = [lambda w: w[0] >= 0.9,\
                         lambda w: w[1] >= 0.9,\
                         lambda w: w[2] >= 0.9]
    self.checkResult(samples, actions, qFuncs, resultConstraints)

  def test_human_world_object_ahead(self):
    """
    Test whether multiple samples are interpreted correctly.
    """
    actions = ['L', 'G', 'R']
    states = [((0.3 * unit, 0), None, (0.3 * unit, 0), None, (100, np.pi / 2)) for unit in xrange(3, 4)]
    samples = [[(state, 'G') for state in states], [(state, 'L') for state in states]]
    qFuncs = modularQFuncs.getHumanWorldQPotentialFuncs()

    resultConstraints = [lambda w: w[0] >= 0.9,\
                         lambda w: w[1] >= 0.9]
    self.checkResult(samples, actions, qFuncs, resultConstraints)

  def test_human_world_confusing_samples(self):
    """
    Make sure IRL doesn't give high weights on any module, if the agent performs inconsistently.
    """
    actions = ['L', 'G', 'R']
    state = ((1, 0), None, (1, 0), None, (1, 0))
    samples = [[(state, 'L'), (state, 'R'), (state, 'G')]]
    qFuncs = modularQFuncs.getHumanWorldQPotentialFuncs()
    
    resultConstraints = [lambda w: w[0] < 0.9 and w[1] < 0.9 and w[2] < 0.9]
    self.checkResult(samples, actions, qFuncs, resultConstraints)

  def checkResult(self, samples, actions, qFuncs, resultConstraints):
    n = len(qFuncs)

    for expIdx in range(len(samples)):
      sln = InverseModularRL(qFuncs)
      sln.getSamples = lambda: samples[expIdx]
      sln.getActions = lambda s: actions
      output = sln.solve()
      x = output.x.tolist()
      w = x[:n]
      d = x[n:]

      judge = resultConstraints[expIdx](x)
      if judge == False:
        # plot weights upon failure
        inverseModularRLExperiments.printWeight(sln, unittest.TestCase.id(self) + '_' + str(expIdx) + '_w.png', d)
        inverseModularRLExperiments.printDiscounter(sln, unittest.TestCase.id(self) + '_' + str(expIdx) + '_d.png', w)

      self.assertTrue(judge, msg="Exp #" + str(expIdx) + " weights: " + str(x))

if __name__ == '__main__':
  unittest.main()