import unittest
from inverseModularRL import InverseModularRL

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