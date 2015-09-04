from cmp import ControlledMarkovProcess

class MachineConfiguration(ControlledMarkovProcess):
  def __init__(self, n, m, rewardFunc, responseFunc, cost=-0.2):
    self.n = n
    self.m = m
    self.cost = cost
    self.rewardFunc = rewardFunc

    self.getResponseTime = 2

    ControlledMarkovProcess.__init__(self, responseFunc)
    
  def reset(self):
    self.state = [0] * self.n

  def isTerminal(self, state):
    return not 0 in state

  def getPossibleActions(self, state):
    """
    Can set non-operated machines
    an action: (i, j) operate machine i to be in config j
    """
    return [(i, j) for j in range(1, self.m+1)\
                   for i in range(self.n) if self.state[i] == 0]

  def getTransitionStatesAndProbs(self, state, action):
    state = self.state[:]
    mch = action[0]
    config = action[1]
    state[mch] = config
    
    return [(state, 1)]
  
  def getReward(self, state):
    if self.isTerminal(state):
      # complete configuration
      sum([self.rewardFunc(i, state[i]) for i in range(self.n)])
    else:
      # incomplete configuration
      return self.cost
