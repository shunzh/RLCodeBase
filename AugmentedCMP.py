from cmp import ControlledMarkovProcess
from QTPAgent import QTPAgent
from mdp import MarkovDecisionProcess

class AugmentedCMP(ControlledMarkovProcess):
  """
  Augment the state, action space of a given CMP, to be an MDP.
  This makes the joint action, query problem to be a pure planning problem.
  Decoration pattern.
  """
  def __init__(self, cmp, rewardSet, initialPhi, queryType, gamma, lLimit):
    """
    Initialize with this cmp 
    """
    MarkovDecisionProcess.__init__(self)
    self.oCmp = cmp
    self.qtpAgent = QTPAgent(cmp, rewardSet, initialPhi, queryType, gamma)
    self.lLimit = lLimit
    self.possiblePsis = set()

    for query in self.oCmp.queries:
      psis = map(lambda (psi, psiProb): tuple(psi), self.qtpAgent.getPossiblePhiAndProbs(query))
      self.possiblePsis.update(psis)
  
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
  
  def doAction(self, action):
    cmpState, psi, l = self.state
    if action in self.cmp.queries:
      # inform the agent a response when asked
      self.oCmp.query(action)
      reward = self.oCmp.cost(action)

      res = self.oCmp.responseCallback()
      assert res != None

      psi = self.qtpAgent.responseToPhi[(action, res)]
      assert psi != 0

      l -= 1
    else:
      cmpState, reward = self.oCmp.doAction(self, action)
      self.oCmp.state = cmpState

    self.state = (cmpState, psi, l)
    
    return (self.state, reward)

  def getReward(self, state):
    cmpState, psi, l = state
    r = self.qtpAgent.getRewardFunc(psi)
    return r(cmpState)
  
  def isTerminal(self, state):
    cmpState, psi, l = state
    return self.oCmp.isTerminal(self, cmpState)
