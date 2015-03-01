import unittest
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
  
  def test_human_world_QPotential(self):
    self.human_world(modularQFuncs.getHumanWorldQPotentialFuncs())

  def test_human_world_discrete(self):
    self.human_world(modularQFuncs.getHumanWorldDiscreteFuncs())

  def human_world(self, qFuncs):
    """
    make some human data, which have clear intentions (to target, or avoid obstacle)
    """
    actions = ['L', 'G', 'R']
    # samples: targest, obstacles, path
    state = ((2, 2), (3, -2), (3, 2), (4, 2), (3, 0))
    samples = [[(state, 'R')], [(state, 'L')], [(state, 'G')]]
    resultConstraints = [lambda w: w[0] >= 0.9,\
                         lambda w: w[1] >= 0.9,\
                         lambda w: w[2] >= 0.9]
    self.checkResult(samples, actions, qFuncs, resultConstraints)

  def checkResult(self, samples, actions, qFuncs, resultConstraints):
    for expIdx in range(len(samples)):
      sln = InverseModularRL(qFuncs)
      sln.getSamples = lambda : samples[expIdx]
      sln.getActions = lambda s: actions
      output = sln.solve()
      w = output.x.tolist()
      
      print expIdx, w
      self.assertTrue(resultConstraints[expIdx](w))

if __name__ == '__main__':
  unittest.main()