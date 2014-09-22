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
    self.eta = 20

  def obj(self, X):
    """
      The objective function to be minimized.

      Args:
        X: parameter vector.
      Return:
        the function value
    """
    states = self.mdp.getStates()
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

  def findWeights(self):
    """
      Find the approporiate weight for each module, by walking through the policy.

      Return:
        optimal weight
    """
    start_pos = np.zeros(3)
    
    # range of weights
    bnds = tuple((0, 1) for x in start_pos)

    # constraints: sum to be 1
    cons = ({'type': 'eq', 'fun': lambda x:  1 - sum(x)})

    w = minimize(self.obj, start_pos, method='SLSQP', bounds=bnds ,constraints=cons)
    return w


def checkPolicyConsistency(states, a, b):
  """
    Check how many policies on the states are consistent with the optimal one.

    Args:
      states: the set of states that we want to compare the policies
      a, b: two agents that we want to compare their policies
    Return:
      Portion of consistent policies
  """
  consistentPolices = 0

  # Walk through each state
  for state in states:
    consistentPolices += (a.getPolicy(state) == b.getPolicy(state))

  return 1.0 * consistentPolices / len(states)


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

    sln = InverseModularRL(a, m, qFuncs)
    w = sln.findWeights()

    # check the consistency between the original optimal policy
    # and the policy predicted by the weights we guessed.
    aHat = modularAgents.ModularAgent(**qLearnOpts)
    aHat.setQFuncs(qFuncs)
    aHat.setWeights(w)
    print checkPolicyConsistency(m.getStates(), a, aHat)


if __name__ == '__main__':
    main()
