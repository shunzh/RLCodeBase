# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math
import pickle
          
class QLearningAgent(ReinforcementAgent):
  """
    Q(lambda)-Learning Agent
    
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
    "You can initialize Q-values here..."
    ReinforcementAgent.__init__(self, **args)

    "*** YOUR CODE HERE ***"
    self.values = util.Counter()
    # initialize e_t
    self.e = util.Counter()
    # default value for lambda
    self.lambdaValue = 0
    self.replace = True
  
  def setLambdaValue(self, lambdaValue):
    self.lambdaValue = lambdaValue

  def getQValue(self, state, action):
    """
      Returns Q(state,action)    
      Should return 0.0 if we never seen
      a state or (state,action) tuple 
    """
    "*** YOUR CODE HERE ***"
    return self.values[state, action]
    
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
    
  def printQValues(self):
    for item, value in self.values.items():
      print 's, a: ', item
      print 'q: ', value
    
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
      maxQValue = max([q_value_func(action) for action in actions])
      optActions = [action for action in actions if q_value_func(action) == maxQValue]
      return random.choice(optActions)
    else:
      return None

    #util.raiseNotDefined()
    
  def getAction(self, state):
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
    action = None
    "*** YOUR CODE HERE ***"
   
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
    delta = reward + self.gamma * self.getValue(nextState) - self.getQValue(state, action)

    if self.replace:
      self.e[state, action] = 1
    else:
      self.e[state, action] += 1

    for state, action in self.e:
      self.values[state, action] += self.alpha * delta * self.e[state, action]
      self.e[state, action] *= self.gamma * self.lambdaValue

  def final(self, state):
    "Called at the end of each game."
    pass


class ReducedQLearningAgent(QLearningAgent):
  """
  Rewrite Q learning agent.

  This is different from ApproximateQAgent, which assumes Q values are linear combination of
  feature values.
  """
  def __init__(self, **args):
    QLearningAgent.__init__(self, **args)

    # Set get state function here.
    # By default, it is an identity function
    self.getState = lambda x : x

  def setStateFilter(self, extractor):
    """
    Set the state filter here, which returns the state representation for learning.
    The default one is an identity function.
    """
    self.getState = extractor

  def getQValue(self, state, action):
    return QLearningAgent.getQValue(self, self.getState(state), action)

  def getValue(self, state):
    return QLearningAgent.getValue(self, self.getState(state))

  def getPolicy(self, state):
    return QLearningAgent.getPolicy(self, self.getState(state))

  def update(self, state, action, nextState, reward):
    return QLearningAgent.update(self, self.getState(state), action, self.getState(nextState), reward)

  def final(self, state):
    return QLearningAgent.final(self, self.getState(state))


class PacmanQAgent(QLearningAgent):
  "Exactly the same as QLearningAgent, but with different default parameters"
  
  def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
    """
    These default parameters can be changed from the pacman.py command line.
    For example, to change the exploration rate, try:
        python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
    
    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
    args['epsilon'] = epsilon
    args['gamma'] = gamma
    args['alpha'] = alpha
    args['numTraining'] = numTraining
    self.index = 0  # This is always Pacman
    QLearningAgent.__init__(self, **args)

  def getAction(self, state):
    """
    Simply calls the getAction method of QLearningAgent and then
    informs parent of action for Pacman.  Do not change or remove this
    method.
    """
    action = QLearningAgent.getAction(self,state)
    self.doAction(state,action)
    return action

    
class ApproximateQAgent(PacmanQAgent):
  """
     ApproximateQLearningAgent
     
     You should only have to overwrite getQValue
     and update.  All other QLearningAgent functions
     should work as is.
  """
  def __init__(self, extractor = IdentityExtractor(), **args):
    self.featExtractor = extractor
    PacmanQAgent.__init__(self, **args)

    # You might want to initialize weights here.
    "*** YOUR CODE HERE ***"
    self.weights = util.Counter()
    
  def getQValue(self, state, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    "*** YOUR CODE HERE ***"
    q = 0.0
    feats = self.featExtractor.getFeatures(state, action)
    if feats != None:
      for feature, value in self.featExtractor.getFeatures(state, action).items():
        q += self.weights[feature] * value
      return q
    else:
      return 0
    
  def update(self, state, action, nextState, reward):
    """
       Should update your weights based on transition  
    """
    "*** YOUR CODE HERE ***"
    correction = (reward + self.gamma * self.getValue(nextState)) - self.getQValue(state, action)

    feats = self.featExtractor.getFeatures(state, action)
    if feats != None:
      for feature, value in feats.items():
        self.weights[feature] += self.alpha * correction * value

  def final(self, state):
    "Called at the end of each game."
    pass
