import environment
import random

class MDPEnvironment(environment.Environment):
  """
  which holds a mdp object.

  #FIXME this is essentially the same for all the domains, consider abstract this.
  """
  def __init__(self, mdp):
    self.mdp = mdp
    self.reset()
            
  def getCurrentState(self):
    return self.state
        
  def getPossibleActions(self, state):        
    return self.mdp.getPossibleActions(state)
        
  def doAction(self, action):
    successors = self.mdp.getTransitionStatesAndProbs(self.state, action) 
    sum = 0.0
    rand = random.random()
    state = self.getCurrentState()
    for nextState, prob in successors:
      sum += prob
      if sum > 1.0:
        raise 'Total transition probability more than one; sample failure.' 
      if rand < sum:
        reward = self.mdp.getReward(state, action, nextState)
        self.state = nextState
        return (nextState, reward)

    raise 'Total transition probability less than one; sample failure.'    
        
  def reset(self):
    self.state = self.mdp.getStartState()

  def isFinal(self):
    return self.mdp.isFinal(self.state)


