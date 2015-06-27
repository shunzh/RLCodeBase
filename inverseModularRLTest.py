"""
This is the test for modular IRL.
Try some simple domains, and human world experiments on both discrete and continuous Q functions.
"""
import unittest
from inverseModularRL import InverseModularRL

class Test(unittest.TestCase):
  methodCandidates = ["CMA-ES", "BFGS"]
  def test_corridor(self):
    """
    Two objectives - going to left or going to right in a corridor.
    """
    actions = ['L', 'R']
    samples = [[(p, actions[0]) for p in range(2)], [(p, actions[1]) for p in range(2)]]
    qFuncs = [lambda s, a, x = None: x[0] if a == actions[0] else 0,\
              lambda s, a, x = None: x[1] if a == actions[1] else 0]
    resultConstraints = [lambda w: w[0] >= 0.99 and w[1] <= 0.01,\
                         lambda w: w[1] >= 0.99 and w[0] <= 0.01]
    self.checkResult(samples, actions, qFuncs, resultConstraints)
    
  def test_three_way(self):
    """
    One state and three actions. One module for taking each action.
    """
    actions = range(3)

    samples = [[(0, action)] for action in actions]
    qFuncs = [lambda s, a, x = None: x[0] if a == actions[0] else 0,\
              lambda s, a, x = None: x[1] if a == actions[1] else 0,\
              lambda s, a, x = None: x[2] if a == actions[2] else 0]
    resultConstraints = [lambda w: w[0] >= 0.99,\
                         lambda w: w[1] >= 0.99,\
                         lambda w: w[2] >= 0.99]
    self.checkResult(samples, actions, qFuncs, resultConstraints)
 
  def checkResult(self, samples, actions, qFuncs, resultConstraints):
    n = len(qFuncs)

    for expIdx in range(len(samples)):
      for methodCandidate in Test.methodCandidates: 
        starts = [0] * n
        bnds = tuple((0, 1000) for _ in xrange(n))

        sln = InverseModularRL(qFuncs, starts, bnds, solver=methodCandidate)

        sln.getSamples = lambda: samples[expIdx]
        sln.getActions = lambda s: actions
        x = sln.solve()

        w = x[:n]
        sumW = sum(w)
        w = [wi / sumW for wi in w]

        judge = resultConstraints[expIdx](w)

        self.assertTrue(judge, msg="Exp #" + str(expIdx) + " parameter: " + str(w) + " with " + methodCandidate)

if __name__ == '__main__':
  unittest.main()