import gridworld as gw
import modularAgents

import numpy as np
from scipy.optimize import fsolve

class InverseModularRL:
  """
    Inverse Reinforcement Learning. From a trained modular agent, find the weights given
    - Q tables of modules
    - Policy function

    This is implemented as
    C. A. Rothkopf and Ballard, D. H.(2013), Modular inverse reinforcement
    learning for visuomotor behavior, Biological Cybernetics,
    107(4),477-490
    http://www.cs.utexas.edu/~dana/Biol_Cyber.pdf
  """

  def __init__(self, agent, mdp, qFuncs):
    """
      Args:
        agent: the modular agent object
        mdp: the hybird environment
        qFuncs: a list of Q functions for all the modules
    """
    self.agent = agent
    self.mdp = mdp
    self.qFuncs = qFuncs

    # confidence on 
    self.eta = 1 

  def findWeights(self):
    """
      Find the approporiate weight for each module, by walking through the policy
    """
    states = self.mdp.getStates()

    def obj(w, lmd):
      """
        The objective function to be minimized.

        Args:
          w: weight vector
          lmd: lambda
      """
      ret = 0

      # Walk through each state
      for state in states:
        optAction = self.agent.getPolicy(state)

        # Update the weights for each module accordingly.
        for moduleIdx in xrange(len(self.qFuncs)):
          ret += self.eta * w[moduleIdx] * self.qFuncs[moduleIdx](state, optAction)

          # denominator
          denom = 0
          actionSet = self.mdp.getPossibleActions(state)
          for action in actionSet:
            denom += np.exp(self.eta * w[moduleIdx] * self.qFuncs[moduleIdx](state, action))
          ret -= np.log(denom)

        ret += lmd * (sum(w) - 1)

      return ret

    def dObj(X):
      """
        Derivative of function obj using numerical method.
      """
      dLambda = np.zeros(len(X))
      h = 1e-3 # this is the step size used in the finite difference.
      for i in range(len(X)):
        # create a vector with a step size in i-th dimension
        dX = np.zeros(len(X))
        dX[i] = h

        dLambda[i] = (obj(X+dX)-obj(X-dX))/(2*h);
      return dLambda

    w = fsolve(dObj, [0] * (len(self.qFuncs) + 1))
    return w


def main():
    """
      Can be called to run pre-specified agent and domain.
    """
    m = gw.getWalkAvoidGrid()

    gridWorldEnv = gw.GridworldEnvironment(m)
    actionFn = lambda state: m.getPossibleActions(state)
    qLearnOpts = {'gamma': 0.9,
                  'alpha': 0.5,
                  'epsilon': 0.3,
                  'actionFn': actionFn}
    a = modularAgents.ModularAgent(**qLearnOpts)
    a.setQFuncs(modularAgents.getObsAvoidFuncs(m))

    sln = InverseModularRL(a, m, qFuncs)
    print sln.findWeights()


if __name__ == '__main__':
    main()
