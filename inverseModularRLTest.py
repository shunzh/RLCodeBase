"""
This is the test for modular IRL.
Try some simple domains, and human world experiments on both discrete and continuous Q functions.
"""
import unittest
from inverseModularRL import InverseModularRL
import modularQFuncs
import util
import continuousWorldDomains
from humanWorld import HumanWorld
import modularAgents

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
  
  def test_human_env(self):
    #TODO
    qFuncs = modularQFuncs.getHumanWorldQPotentialFuncs()[:2]
    n = len(qFuncs)
    x = [1, -1] + [.5, .5] + [0, 0.5]

    init = continuousWorldDomains.loadFromMat('miniRes25.mat', 0)
    mdp = HumanWorld(init)
    actionFn = lambda state: mdp.getPossibleActions(state)
    qLearnOpts = {'gamma': 0.9,
                  'alpha': 0.5,
                  'epsilon': 0,
                  'actionFn': actionFn}
    agent = modularAgents.ModularAgent(**qLearnOpts)
    agent.setParameters(x)

    starts = [0] * n + [0.5] * n
    margin = 0.1
    bnds = ((0, 1000), (-1000, 0))\
         + tuple((0 + margin, 1 - margin) for _ in range(n))\
         + ((0, 10), (0, 10))

    sln = InverseModularRL(qFuncs, starts, bnds, solver="CMA-ES")
    sln.setSamplesFromMdp(mdp, agent)

    sln.solve()
 
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