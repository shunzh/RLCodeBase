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
    self.eta = 0.9

  def findWeights(self):
    """
      Find the approporiate weight for each module, by walking through the policy
    """
    states = self.mdp.getStates()

    def obj(X):
      """
        The objective function to be minimized.

        Args:
          X: a vector of length len(qFuncs) + 2.
             The first len(qFuncs) elements are the weights for corresponding module.
             The last two elements are Lagrange multipiers.
      """
      w = X[:-1]
      lmd1 = X[-1]
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

      ret += lmd1 * (sum(w) - 1)

      # doesn't converge while adding this constraint o_o
      #ret += - lmd2 * sum([np.absolute(wi) for wi in w])

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
    # environment, an mdp object
    m = gw.getLargeWalkAvoidGrid()

    gridWorldEnv = gw.GridworldEnvironment(m)
    actionFn = lambda state: m.getPossibleActions(state)
    qLearnOpts = {'gamma': 0.9,
                  'alpha': 0.5,
                  'epsilon': 0.3,
                  'actionFn': actionFn}
    # modular agent
    a = modularAgents.ModularAgent(**qLearnOpts)

    qFuncs = modularAgents.getObsAvoidFuncs(m)
    # set the weights and corresponding q-functions for its sub-mdps
    # note that the modular agent is able to determine the optimal policy based on these
    a.setQFuncs(qFuncs)
    a.setWeights([0, 1])

    sln = InverseModularRL(a, m, qFuncs)
    print sln.findWeights()


if __name__ == '__main__':
    main()
