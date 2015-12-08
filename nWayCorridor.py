from robotNavigation import RobotNavigation

class NWayCorridor(RobotNavigation):
  def __init__(self, queries, trueReward, gamma, responseTime, width, height, horizon, terminalReward):
    RobotNavigation.__init__(self, queries, trueReward, gamma, responseTime, width, height, horizon, terminalReward)
  
  def reset(self):
    self.state = (self.width / 2, 0)
  
  def isTerminal(self, state):
    return state[1] == self.height - 1
  
  def getPossibleActions(self, state):
    actions = [(0, 1)]

    if state[1] == 0:
      actions = RobotNavigation.getPossibleActions(self, state)
      #actions.append((1, 0)
    
    return actions