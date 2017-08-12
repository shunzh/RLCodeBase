from mdp import MarkovDecisionProcess
import itertools

class FactoredMDP(MarkovDecisionProcess):
  def __init__(self, sSets, a, rFunc, tFunc, isTerminal, s0, isTerminal, gamma):
    MarkovDecisionProcess.__init__(self)
    
    # these functions can be constructed easily
    self.getStates = lambda: itertools.product(*sSets)
    self.getPossibleActions = lambda state: a
    self.isTerminal = isTerminal
    # factored reward function
    #self.getReward = lambda state, action: sum(r(s, a) for s, r in zip(state, rFunc))
    # nonfactored reward function
    self.getReward = rFunc
    self.getStartState = lambda: s0
    self.isTerminal = isTerminal
    self.gamma = gamma

    # keep this information for now
    self.tFunc = tFunc
  
  def getTransitionStatesAndProbs(self, state, action):
    """
    t(s, a, s') = \prod t_i(s, a, s_i)
    
    #FIXME assume deterministic transitions for now to make the life easier!
    """
    sp = (t(s, action) for s, t in zip(state, self.tFunc))
    return [(sp, 1)]

  
class ConstrainedFactoredMDP(FactoredMDP):
  def __init__(self, sSets, cIndices, a, rFunc, tFunc, isTerminal, s0, isTerminal, gamma):
    """
    cSets specify a set of states whose values should not be changed
    """
    FactoredMDP.__init__(self, sSets, a, rFunc, tFunc, isTerminal, s0, isTerminal, gamma)
    self.cIndices = cIndices
