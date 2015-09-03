from cmp import ControlledMarkovProcess

class MachineConfiguration(ControlledMarkovProcess):
  def __init__(self, n, m):
    ControlledMarkovProcess.__init__(self)

    self.n = n
    self.m = m
  
  def getStartState(self):
    return [0] * self.n
  
  def getResponseTime(self, state):
    return 2