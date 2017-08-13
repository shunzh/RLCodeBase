from QTPAgent import QTPAgent
from mdp import MarkovDecisionProcess
import util

class AugmentedCMP(MarkovDecisionProcess):
  """
  Augment the state, action space of a given CMP, to be an MDP.
  This makes the joint action, query problem to be a pure planning problem.
  Decoration pattern.
  """
  def __init__(self, cmp, rewardSet, initialPsi, queryType, gamma, lLimit, awina=False):
    """
    Initialize with this cmp 
    """
    self.oCmp = cmp
    self.qtpAgent = QTPAgent(cmp, rewardSet, initialPsi, queryType, gamma)
    self.initialPsi = initialPsi
    self.lLimit = lLimit
    self.queryType = queryType
    self.possiblePsis = set()
    self.awina = awina

    for query in self.oCmp.queries:
      psis = map(lambda (psi, psiProb): tuple(psi), self.qtpAgent.getPossiblePhiAndProbs(query))
      self.possiblePsis.update(psis)
    
    self.viInitial = util.Counter()
    if awina:
      self.viSet = util.Counter()
      for psi in self.possiblePsis:
        vi = self.qtpAgent.getVIAgent(psi)
        for state in self.oCmp.getStates():
          self.viInitial[(state, psi, 0)] = vi.getValue(state)
        self.viSet[psi] = vi

      self.eliminateQueries = util.Counter()
      for state in self.oCmp.getStates():
        self.eliminateQueries[state] = []
        for query in self.oCmp.queries:
          policySet = set()
          policySet.update(self.oCmp.getPossibleActions(state))
          psis = self.qtpAgent.getPossiblePhiAndProbs(query)
          for psi in psis:
            policySet.intersection_update(self.viSet[psi[0]].getPolicies(state))
          if len(policySet) > 0:
            self.eliminateQueries[state].append(query)
  
    MarkovDecisionProcess.__init__(self)

  def reset(self):
    self.oCmp.reset()
    self.state = (self.oCmp.state, self.initialPsi, self.lLimit)
  
  def getVIInitial(self):
    return self.viInitial
  
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
      if self.awina:
        queries = list(set(self.oCmp.queries) - set(self.eliminateQueries[cmpState]))
      else:
        queries = self.oCmp.queries
      actions = decorate(actions, 'a') + decorate(queries, 'q')
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
