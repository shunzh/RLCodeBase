from cmp import ControlledMarkovProcess

class MachineConfiguration(ControlledMarkovProcess):
  def __init__(self, n, m, trueReward, queries, gamma, responseTime):
    self.n = n
    self.m = m

    ControlledMarkovProcess.__init__(self, queries, trueReward, gamma, responseTime)
    
  def getStates(self):
    configs = range(self.m + 1)
    # construct the state space without selection
    l = [[]]
    for _ in range(self.n):
      newL = []
      for item in l:
        newL += [item + [i] for i in configs]
      l = newL
      
    l = map(tuple, l)
    
    # add possible selecting status
    selectionStates = []
    for item in l:
      for idx in range(self.n):
        if item[idx] == 0:
          newItem = list(item[:])
          newItem[idx] = 'S'
          selectionStates.append(tuple(newItem))
    l += selectionStates

    return l

  def cost(self, query):
    return 0
  
  def reset(self):
    self.state = (0,) * self.n

  def isTerminal(self, state):
    # no selection pending, ever compnent is configured
    return all([type(c) is int and c > 0 for c in state])

  def getPossibleActions(self, state):
    """
    Can set non-operated machines
    an action: (i, j) operate machine i to be in config j
    """
    actions = ['Wait']
    
    if 'S' in state:
      # an action is selected, configuration actions are available
      i = state.index('S')
      actions += [(i, j) for j in range(1, self.m+1)]
    else:
      # choose a component for configuration
      actions += [(i, 'S') for i in range(self.n) if state[i] == 0]
    
    return actions

  def getTransitionStatesAndProbs(self, state, action):
    assert action in self.getPossibleActions(state)

    if action == 'Wait':
      # means stay
      return [(state, 1)]
    else:
      # make a deep copy
      state = list(state[:])

      mch = action[0]
      config = action[1]
      state[mch] = config
      
      return [(tuple(state), 1)]