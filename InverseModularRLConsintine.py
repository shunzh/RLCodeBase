from inverseModularRL import InverseModularRL
import numpy as np

class InverseModularRLConsintine(InverseModularRL):
  def __init__(self, qFuncs, starts, bnds,\
               decorator = lambda x: x,\
               solver = "BFGS"):
    InverseModularRL.__init__(self, qFuncs, starts, bnds, decorator, solver=solver, eta=10)

  def obj(self, X):
    """
    !!!!!!!!!!!!!!ALERT!!!!!!!!!!!!!!
    This is a wrong formula and should only be used for comparison
    """
    X = self.decorator(X)
    reg = self.regular(X)
    ret = 0

    # replay the process
    for state, optAction in self.getSamples():
      term = 0

      # Update the weights for each module accordingly.
      for moduleIdx in xrange(len(self.qFuncs)):
        term += self.eta * self.qFuncs[moduleIdx](state, optAction, X)

        # denominator
        denom = 0
        actionSet = self.getActions(state)
        for action in actionSet:
          denom += np.exp(self.eta * self.qFuncs[moduleIdx](state, action, X))

        term -= np.log(denom)

      ret += term

    # This is to be minimized, take the negative.
    return - ret