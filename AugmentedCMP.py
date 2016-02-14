from cmp import ControlledMarkovProcess
import util
from QTPAgent import QTPAgent

class AugmentedCMP(ControlledMarkovProcess):
  """
  Augment the state, action space of a given CMP
  to make the joint action, query problem to be a pure planning problem.
  Decoration pattern.
  """
  def __init__(self, cmp, rewardSet, initialPhi, queryType, gamma, lLimit):
    """
    Initialize with this cmp 
    """
    self.oCmp = cmp
    self.qtpAgent = QTPAgent(cmp, rewardSet, initialPhi, queryType, gamma)
    self.lLimit = lLimit

    # possiblePsis todo
    for query in self.oCmp.queries:
      self.qtpAgent.getPossiblePhiAndProbs(query)
  
  def reset(self):
    self.oCmp.reset()
    self.state = self.oCmp.state
  
  def getStates(self):
    cmpStates = self.oCmp.getStates(self)
    states = []

    for l in xrange(self.lLimit):
      for psi in self.possiblePsis:
        for cmpState in cmpStates:
          states.append((cmpState, psi, l))
    return states
  
  def getPossibleActions(self, state):
    cmpState, psi, l = state
    actions = self.oCmp.getPossibleActions(cmpState)
    if l > 0:
      return actions + self.oCmp.queries
    else:
      return actions
  
  def getTransitionStatesAndProbs(self, state, action):
    cmpState, psi, l = state
    if action in self.cmp.queries:
      # transition in reward-knowledge state
      possiblePsiProbs = self.qtpAgent.getPossiblePhiAndProbs(action)
      return map(lambda (psi, psiProb): ((cmpState, psi, l-1), psiProb), possiblePsiProbs)
    else:
      # transition in decision making
      cmpStates = self.oCmp.getTransitionStatesAndProbs(cmpState, action)
      return map(lambda (s, prob): ((s, psi, l), prob), cmpStates)
  
  def getReward(self, state):
    cmpState, psi, l = state
    r = self.qtpAgent.getRewardFunc(psi)
    return r(cmpState)
  
  def isTerminal(self, state):
    cmpState, psi, l = state
    return self.oCmp.isTerminal(self, cmpState)