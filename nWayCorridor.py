from robotNavigation import RobotNavigation

class NWayCorridor(RobotNavigation):
  def __init__(self, queries, trueReward, gamma, responseTime, width, height, horizon, terminalReward):
    self.corridorLen = height - 2
    RobotNavigation.__init__(self, queries, trueReward, gamma, responseTime, width, height, horizon, terminalReward)
  
  def reset(self):
    self.state = (self.width / 2, 0)
  
  def isTerminal(self, state):
    return state[1] == self.height - 1
  
  def getPossibleActions(self, state):
    if state[1] > self.height - self.corridorLen:
      return [(0, 1)]
    else:
      return RobotNavigation.getPossibleActions(self, state)