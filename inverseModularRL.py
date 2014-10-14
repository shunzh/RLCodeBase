import modularAgents

import numpy as np
from scipy.optimize import minimize

import sys

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

  def __init__(self, agent, mdp, recorder, qFuncs):
    """
      Args:
        agent: the modular agent object
        mdp: the hybird environment
        qFuncs: a list of Q functions for all the modules
    """
    self.agent = agent
    self.mdp = mdp
    self.recorder = recorder
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
    w = X
    ret = 0

    # replay the process
    for (state, optAction) in self.recorder:
      term = 0

      # Update the weights for each module accordingly.
      for moduleIdx in xrange(len(self.qFuncs)):
        term += self.eta * w[moduleIdx] * self.qFuncs[moduleIdx](state, optAction)

        # denominator
        denom = 0
        actionSet = self.mdp.getPossibleActions(state)
        for action in actionSet:
          denom += np.exp(self.eta * w[moduleIdx] * self.qFuncs[moduleIdx](state, action))
        term -= np.log(denom)

      ret += term

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


def checkPolicyConsistency(recorder, a, b):
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
  for (state, action) in recorder:
    print state, a.getPolicy(state), action
    consistentPolices += int(a.getPolicy(state) == b.getPolicy(state))

  return 1.0 * consistentPolices / len(recorder)


def getWeightDistance(w1, w2):
  """
    Return:
      ||w1 - w2||_2
  """
  assert len(w1) == len(w2)

  return np.linalg.norm([w1[i] - w2[i] for i in range(len(w1))])


def main():
  """
    Can be called to run pre-specified agent and domain.
  """
  # environment, an mdp object FIXME
  #import gridworld as gw
  #m = gw.getLargeWalkAvoidGrid(0.4)
  #gridWorldEnv = gw.GridworldEnvironment(m)
  
  import continuousWorld as cw
  #init = cw.loadFromMat('miniRes25.mat', 0)
  init = cw.toyDomain()
  m = cw.ContinuousWorld(init)
  env = cw.ContinuousEnvironment(m)

  actionFn = lambda state: m.getPossibleActions(state)
  qLearnOpts = {'gamma': 0.9,
                'alpha': 0.5,
                'epsilon': 0,
                'actionFn': actionFn}
  # modular agent
  a = modularAgents.ModularAgent(**qLearnOpts)

  if len(sys.argv) > 1:
    # user wants to set weights themselves
    w = map(float, sys.argv[1:])
    a.setWeights(w)

  #qFuncs = modularAgents.getObsAvoidFuncs(m)
  qFuncs = modularAgents.getContinuousWorldFuncs(m)
  # set the weights and corresponding q-functions for its sub-mdps
  # note that the modular agent is able to determine the optimal policy based on these
  a.setQFuncs(qFuncs)

  print "Ready for simulation"

  recorder = []
  noneFunc = lambda *x: None
  cw.runEpisode(a, env, 0.9, a.getAction, noneFunc, noneFunc, noneFunc, 1, recorder)

  print "Simulation done. Recover from samples.."

  sln = InverseModularRL(a, m, recorder, qFuncs)
  output = sln.findWeights()
  w = output.x.tolist()
  w = map(lambda _: round(_, 5), w) # avoid weird numerical problem

  print "IRL done."
  print "Weight: ", w

  # re-initialize
  m = cw.ContinuousWorld(init)
  qFuncs = modularAgents.getContinuousWorldFuncs(m)
  a = modularAgents.ModularAgent(**qLearnOpts)
  a.setQFuncs(qFuncs)

  # check the consistency between the original optimal policy
  # and the policy predicted by the weights we guessed.
  aHat = modularAgents.ModularAgent(**qLearnOpts)
  aHat.setQFuncs(qFuncs)
  aHat.setWeights(w) # get the weights in the result

  # print for experiments
  print checkPolicyConsistency(recorder, a, aHat)
  #print checkPolicyConsistency(recorder, a, aHat)
  print getWeightDistance(a.getWeights(), w)


if __name__ == '__main__':
  main()
