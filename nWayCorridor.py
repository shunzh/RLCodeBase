from robotNavigation import RobotNavigation
import config

class NWayCorridor(RobotNavigation):
  def __init__(self, queries, trueReward, gamma, responseTime, width, height, horizon, terminalReward):
    self.corridorWidth = config.PARAMETER # this is passed as the domain parameter
    RobotNavigation.__init__(self, queries, trueReward, gamma, responseTime, width, height, horizon, terminalReward)
  
  def reset(self):
    self.state = (self.width / 2, 0)
  
  def isTerminal(self, state):
    return state[1] == self.height - 1
  
  def getPossibleActions(self, state):
    actions = [(0, 1)]

    if state[1] < 3:
      actions = RobotNavigation.getPossibleActions(self, state)
      if state[1] > 0:
        if state[0] % self.corridorWidth == 0:
          actions.remove((-1, 0))
        if state[0] % self.corridorWidth == self.corridorWidth - 1:
          actions.remove((1, 0))
    
    return actions