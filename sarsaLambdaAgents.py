# sarsaLambdaAgents.py

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math
          
class SarsaLambdaAgent(ReinforcementAgent):
  """
    SarsaLearningAgent
    
    Functions you should fill in:
      - getQValue
      - getAction
      - getValue
      - getPolicy
      - update
      
    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.gamma (discount rate)
    
    Functions you should use
      - self.getLegalActions(state) 
        which returns legal actions
        for a state
  """
  def __init__(self, **args):
    "initialize..."
    ReinforcementAgent.__init__(self, **args)

    "*** YOUR CODE HERE ***"
    # value counter
    self.values = util.Counter()
    # initialize e_t
    self.e = util.Counter()
    # keeping the action of next state
    self.nextAction = None
  
  def getQValue(self, state, action):
    """
      Returns Q(state,action)    
      Should return 0.0 if we never seen
      a state or (state,action) tuple 
    """
    "*** YOUR CODE HERE ***"
    return self.values[state, action]
    
    #util.raiseNotDefined()
  
    
  def getValue(self, state):
    """
      Returns max_action Q(state,action)        
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    "*** YOUR CODE HERE ***"
    bestAction = self.getPolicy(state)
    if bestAction: 
      return self.getQValue(state, bestAction)
    else: 
      return 0.0
   #util.raiseNotDefined()
    
  def getPolicy(self, state):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    "*** YOUR CODE HERE ***"
    actions = self.getLegalActions(state)
    if actions: 
      q_value_func = lambda action: self.getQValue(state, action)
      return max(actions, key=q_value_func)
    else:
      return None

    #util.raiseNotDefined()
    
  def getAction(self, state):
    """
      For SARSA, the action should be determined in the previous state (as action')
      If so, return it. Otherwise, calculate it.
    """
    if self.nextAction:
      return self.nextAction
    else:
      return self.calculateAction(state)
   
  def calculateAction(self, state):
    """
      Compute the action to take in the current state.  With
      probability self.epsilon, we should take a random action and
      take the best policy action otherwise.  Note that if there are
      no legal actions, which is the case at the terminal state, you
      should choose None as the action.
    
      HINT: You might want to use util.flipCoin(prob)
      HINT: To pick randomly from a list, use random.choice(list)
    """  
    # Pick Action
    legalActions = self.getLegalActions(state)
    "*** YOUR CODE HERE ***"
    # when there it's now the TERMINAL_STATE
    if len(legalActions) == 0:
      return None

    if util.flipCoin(self.epsilon): 
      action = random.choice(legalActions)
    else: 
      action = self.getPolicy(state)

    return action
  
  def update(self, state, action, nextState, reward):
    """
      The parent class calls this to observe a 
      state = action => nextState and reward transition.
      You should do your Q-Value update here
      
      NOTE: You should never call this function,
      it will be called on your behalf
    """
    "*** YOUR CODE HERE ***"
		# we need next action for SARSA
    self.nextAction = self.calculateAction(nextState)
    
    delta = reward + self.gamma * self.getQValue(nextState, self.nextAction) - self.getQValue(state, action)
    if self.replace:
      self.e[state, action] = 1
    else:
      self.e[state, action] += 1

    for state, action in self.values:
      self.values[state, action] += self.alpha * delta * self.e[state, action]
      self.e[state, action] *= self.gamma * self.lambdaValue

    if self.nextAction == None:
      # clear eligibility trace when this episode ends
      self.e = Counter()
