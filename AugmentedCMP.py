from QTPAgent import QTPAgent
from mdp import MarkovDecisionProcess

class AugmentedCMP(MarkovDecisionProcess):
  """
  Augment the state, action space of a given CMP, to be an MDP.
  This makes the joint action, query problem to be a pure planning problem.
  Decoration pattern.
  """
  def __init__(self, cmp, rewardSet, initialPsi, queryType, gamma, lLimit):
    """
    Initialize with this cmp 
    """
    self.oCmp = cmp
    self.qtpAgent = QTPAgent(cmp, rewardSet, initialPsi, queryType, gamma)
    self.initialPsi = initialPsi
    self.lLimit = lLimit
    self.queryType = queryType
    self.possiblePsis = set()

    for query in self.oCmp.queries:
      psis = map(lambda (psi, psiProb): tuple(psi), self.qtpAgent.getPossiblePhiAndProbs(query))
      self.possiblePsis.update(psis)

    MarkovDecisionProcess.__init__(self)
  
  def reset(self):
    self.oCmp.reset()
    self.state = (self.oCmp.state, self.initialPsi, self.lLimit)
  
  def getStates(self):
    cmpStates = self.oCmp.getStates()
    states = []

    for l in xrange(self.lLimit):
      for psi in self.possiblePsis:
        for cmpState in cmpStates:
          states.append((cmpState, tuple(psi), l))
    
    for cmpState in cmpStates:
      states.append((cmpState, tuple(self.initialPsi), self.lLimit))

    return states
  
  def getPossibleActions(self, state):
    # need to add a prefix in each action to distinguish
    # whether it is a physical action or query action
    decorate = lambda l, decorator: map(lambda _: (decorator, _), l)
    cmpState, psi, l = state
    actions = self.oCmp.getPossibleActions(cmpState)
    if l > 0:
      actions = decorate(actions, 'a') + decorate(self.oCmp.queries, 'q')
    else:
      actions = decorate(actions, 'a')
    
    return actions

  def getTransitionStatesAndProbs(self, state, action):
    cmpState, psi, l = state
    dec, act = action
    if dec == 'q':
      # transition in reward-knowledge state
      possiblePsiProbs = self.qtpAgent.getPossiblePhiAndProbs(act)
      return map(lambda (psi, psiProb): ((cmpState, tuple(psi), l-1), psiProb), possiblePsiProbs)
    else:
      # transition in decision making
      cmpStates = self.oCmp.getTransitionStatesAndProbs(cmpState, act)
      return map(lambda (s, prob): ((s, tuple(psi), l), prob), cmpStates)
  
  def doAction(self, action):
    cmpState, psi, l = self.state
    dec, act = action
    if dec == 'q':
      # inform the agent a response when asked
      self.oCmp.query((self.queryType, act))
      reward = self.oCmp.cost(act)

      res = self.oCmp.responseCallback()
      print 'res', res
      assert res != None

      psi = self.qtpAgent.responseToPhi[(act, res)]
      assert psi != 0

      l -= 1
    else:
      cmpState, reward = self.oCmp.doAction(act)
      self.oCmp.state = cmpState

    self.state = (cmpState, tuple(psi), l)
    
    return (self.state, reward)

  def getReward(self, state):
    cmpState, psi, l = state
    r = self.qtpAgent.getRewardFunc(psi)
    return r(cmpState)
  
  def isTerminal(self, state):
    cmpState, psi, l = state
    return self.oCmp.isTerminal(cmpState)
