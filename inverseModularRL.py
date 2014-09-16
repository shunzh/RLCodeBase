import gridworld as gw
import modularAgents

import numpy as np
from scipy.optimize import minimize

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
      w = X
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

      # This is to be minimized, take the negative.
      return - ret

    start_pos = np.zeros(3)
    
    # range of weights
    bnds = tuple((0, 1) for x in start_pos)

    # constraints: sum to be 1
    cons = ({'type': 'eq', 'fun': lambda x:  1 - sum(x)})

    w = minimize(obj, start_pos, method='SLSQP', bounds=bnds ,constraints=cons)
    return w


def main():
    """
      Can be called to run pre-specified agent and domain.
    """
    # environment, an mdp object
    m = gw.getWalkAvoidGrid()

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
    a.setWeights([0, 1, 0])

    sln = InverseModularRL(a, m, qFuncs)
    print sln.findWeights()


if __name__ == '__main__':
    main()
