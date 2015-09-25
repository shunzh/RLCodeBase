from cmp import ControlledMarkovProcess

# assume two-dimensional corridor
SIDES = ['L', 'R']

class WindyCorridor(ControlledMarkovProcess):
  def __init__(self, queries, trueReward, gamma, responseTime, interLength, interNum, circular=False):
    self.interLength = interLength
    self.interNum = interNum
    self.circular = circular

    ControlledMarkovProcess.__init__(self, queries, trueReward, gamma, responseTime)
  
  def cost(self, query):
    return 0

  def reset(self):
    # initial state: this far to the first intersection
    self.state = (0, 0)

  def isTerminal(self, state):
    return state == 'T'

  def getStates(self):
    states = [(interId, interDist) for interId in xrange(self.interNum)\
                                   for interDist in xrange(self.interLength)]
    states+= [(interId, side) for interId in xrange(self.interNum)\
                              for side in SIDES]
    states+= ['T'] # terminal state
    return states

  def getPossibleActions(self, state):
    if state[1] == self.interLength - 1:
      # when reaching intersection, can choose to exploit two side states, or go straight ahead
      return ['L', 'R', 'G']
    else:
      # otherwise just continue drifting
      return ['G']

  def getTransitionStatesAndProbs(self, state, action):
    assert action in self.getPossibleActions(state), "Unexpected action %r from state %r" % (action, state)
    
    # terminal state
    if state == 'T': return []

    interId, interDist = state

    if action == 'G':
      if interId == self.interNum - 1 and (interDist == self.interLength - 1 or interDist in SIDES):
        # end of simulation?
        if self.circular: nextState = (0, 0)
        else: nextState = 'T'
      elif type(interDist) is int and interDist < self.interLength - 1:
        nextState = (interId, interDist + 1)
      else:
        # move to the next segment
        nextState = (interId + 1, 0)
    else:
      # side states
      nextState = (interId, action)
    
    # don't make up a state!
    assert nextState in self.getStates(), "Unexpected next state %r from state %r" % (nextState, state)

    # deterministic transition
    return [(nextState, 1)]