from cmp import ControlledMarkovProcess

class BranchingCorridor(ControlledMarkovProcess):
  def __init__(self, queries, trueReward, gamma, responseTime, branches, length):
    self.branches = branches
    self.length = length
    ControlledMarkovProcess.__init__(self, queries, trueReward, gamma, responseTime)

  def cost(self, query):
    return 0

  def getStates(self):
    states = [(x, y) for x in xrange(self.branches)\
                     for y in xrange(self.length)]
    return states

  def reset(self):
    self.state = (0, 0)
    
  def isTerminal(self, state):
    return state[1] == self.length

  def getPossibleActions(self, state):
    # actions are coordinate diff
    if state == (0, 0):
      actions = range(self.branches)
    else:
      # no choices once a branch is chosen
      actions = ['Proceed']
   
    return actions
  
  def getTransitionStatesAndProbs(self, state, action):
    if action in range(self.branches):
      newState = (action, 0)
    elif action == 'Proceed':
      b, y = state
      newState = (b, y+1)
    else:
      raise Exception("Unknown action")

    return [(newState, 1)]
  