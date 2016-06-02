from tabularNavigation import TabularNavigation

class NWayCorridor(TabularNavigation):
  def __init__(self, queries, trueReward, gamma, responseTime, width, height, horizon, terminalReward):
    self.corridorWidth = 1
    TabularNavigation.__init__(self, queries, trueReward, gamma, responseTime, width, height, horizon, terminalReward)
  
  def reset(self):
    self.state = (0, 0)
  
  def isTerminal(self, state):
    return state[1] == self.height - 1
  
  def getPossibleActions(self, state):
    if state == (0, 0):
      actions = [(x, 1) for x in xrange(self.width)]
    else:
      if state[1] < 5:
        actions = TabularNavigation.getPossibleActions(self, state)
      else:
        actions = [(0, 1)]
      
    return actions