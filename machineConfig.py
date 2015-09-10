from cmp import ControlledMarkovProcess

class MachineConfiguration(ControlledMarkovProcess):
  def __init__(self, n, m, trueReward, queries):
    self.n = n
    self.m = m

    self.responseTime = 2

    ControlledMarkovProcess.__init__(self, queries, trueReward)
    
  def getStates(self):
    configs = range(self.m + 1)
    l = [[]]
    for _ in range(self.n):
      newL = []
      for item in l:
        newL += [item + [i] for i in configs]
      l = newL
      
    l = map(tuple, l)
    return l

  def cost(self, query):
    return 0
  
  def reset(self):
    self.state = (0,) * self.n

  def isTerminal(self, state):
    return 0 in state

  def getPossibleActions(self, state):
    """
    Can set non-operated machines
    an action: (i, j) operate machine i to be in config j
    """
    return [None]\
         + [(i, j) for j in range(1, self.m+1)\
                   for i in range(self.n) if self.state[i] == 0]

  def getTransitionStatesAndProbs(self, state, action):
    if action == None:
      # means stay
      return [(state, 1)]
    else:
      # make a deep copy
      state = list(self.state[:])

      mch = action[0]
      config = action[1]
      state[mch] = config
      
      return [(tuple(state), 1)]