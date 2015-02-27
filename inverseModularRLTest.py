import unittest
from inverseModularRL import InverseModularRL

class Test(unittest.TestCase):
  def test_corridor(self):
    """
    set some obvious behavior to check the correctness of the algorithm.
    """
    actions = ['L', 'R']
    samples = [[(p, actions[0]) for p in range(10)], [(p, actions[1]) for p in range(10)]]
    qFuncs = [lambda s, a, d = None: 1 if a == actions[0] else 0,\
              lambda s, a, d = None: 1 if a == actions[1] else 0]
    resultConstraints = [lambda w: w[0] >= 0.99 and w[1] <= 0.01,\
                        lambda w: w[1] >= 0.99 and w[0] <= 0.01]
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