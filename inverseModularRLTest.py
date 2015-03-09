"""
This is the test for modular IRL.
Try some simple domains, and human world experiments on both discrete and continuous Q functions.
"""
import unittest
from inverseModularRL import InverseModularRL
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
 
  def checkResult(self, samples, actions, qFuncs, resultConstraints):
    n = len(qFuncs)

    for expIdx in range(len(samples)):
      sln = InverseModularRL(qFuncs)
      sln.getSamples = lambda: samples[expIdx]
      sln.getActions = lambda s: actions
      x = sln.solve()
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