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

  def __init__(self, qFuncs, eta = 1):
    """
      Args:
        qFuncs: a list of Q functions for all the modules
    """
    self.qFuncs = qFuncs

    # enable if learning discounters as well
    self.learnDiscounter = True
    # confidence
    self.eta = eta

  def setSamplesFromMdp(self, mdp, agent):
    """
    One way to set the samples.
    Read form mdp and get the policies of states.

    Set self.getSamples and self.getActions here.
    """
    states = mdp.getStates()
    self.getSamples = lambda : [(state, agent.getPolicy(state)) for state in states]
    self.getActions = lambda s : mdp.getPossibleActions(s)

  def setSamples(self, samples):
    """
    Read from subj*.parsed.mat file.

    Set self.getSamples and self.getActions here.
    """
    self.getSamples = lambda : samples
    # FIXME overfit
    self.getActions = lambda s : ['L', 'R', 'G']

  def printSamples(self):
    samples = self.getSamples()

    for state, action in samples:
      print state
      print action
    
  def obj(self, X):
    """
      The objective function to be minimized.

      Args:
        X: parameter vector.
      Return:
        the function value
    """
    w = X[:self.n] # weights
    
    if self.learnDiscounter:
      d = X[self.n:]
    else:
      # use default discounters if not learning
      d = [.8] * self.n

    ret = 0

    # replay the process
    for state, optAction in self.getSamples():
      def computeQValue(state, action):
        # s, a -> q(s, a)
        return sum([w[moduleIdx] * self.qFuncs[moduleIdx](state, action, d) for moduleIdx in xrange(len(self.qFuncs))])
      def qToPower(v):
        # v -> exp(eta * v)
        return np.exp(self.eta * v)

      actionSet = self.getActions(state)
      qValues = {action: computeQValue(state, action) for action in actionSet}

      # both numerator and denominator raise to the power of e
      # numerator: optimal action
      ret += qValues[optAction]
      # denominator: all the actions
      ret -= np.log(sum([qToPower(qValues[action]) for action in actionSet]))

    # This is to be minimized, take the negative.
    return - ret

  def solve(self):
    """
      Find the appropriate weight and discounter for each module, by walking through the policy.

      Return:
        optimal weight and discounter, in one vector
    """
    self.n = len(self.qFuncs)

    # range of weights: (0, 1)
    bnds = tuple((0, 1) for _ in range(self.n))
    if self.learnDiscounter:
      # range of discounters
      bnds += tuple((0.01, 0.99) for _ in range(self.n))

    start_pos = np.zeros(len(bnds))

    # constraints: weights must sum to 1
    cons = ({'type': 'eq', 'fun': lambda x:  1 - sum(x[:self.n])})

    x = minimize(self.obj, start_pos, method='SLSQP', bounds=bnds ,constraints=cons)
    return x