from cmp import ControlledMarkovProcess

class AugmentedCMP(ControlledMarkovProcess):
  """
  Augment the state, action space of a given CMP
  to make the joint action, query problem to be a pure planning problem.
  Decoration pattern.
  """
  def __init__(self, cmp):
    """
    Initialize with this cmp 
    """
    self.originalCmp = cmp
  
  def reset(self):
    self.originalCmp.reset()
    self.state = self.originalCmp.state
  
  def getStates(self):
    states = self.originalCmp.getStates(self)
  
  def getPossibleActions(self, state):
    actions = self.originalCmp.getPossibleActions(self, state)
  
  def getTransitionStatesAndProbs(self, state, action):
    self.originalCmp.getTransitionStatesAndProbs(self, state, action)
  
  def getReward(self, state):