# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import os
from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
import numpy as np

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
    
    # a list of temporal difference errors
    self.deltas = []
  
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
    
  def getPolicyProbability(self, state, action):
    """
    Return the probability of choosing action on a state.
    Assume softmax.

    arg: action
    return: the probability of choosing such action
            e^Q(s, a) / sum(e^Q(s, b) for all b)
    """
    actions = self.getLegalActions(state)
    exps = {action: np.exp(self.getQValue(state, action)) for action in actions}
    return exps[action] / sum(exps.values())

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
    
    self.deltas.append(abs(delta))

  def final(self, state):
    # clear deltas after an episode
    self.deltas = []


class ReducedQLearningAgent(QLearningAgent):
  """
  Wrap the Q learning agent with a state filter, which reduces the state space.
  
  State: S -> S', where S' is significantly smaller.
  One usage is mapping continuous state space into discrete bins.
  """
  def __init__(self, **args):
    QLearningAgent.__init__(self, **args)

    # Set get state function here.
    # By default, it is an identity function
    self.getState = lambda x : x

  def setValues(self, filename):
    """
      Set initial weights, if we have it
    """
    if os.path.isfile(filename):
      import pickle
      self.values = pickle.load(open(filename))
    else:
      import warnings
      warnings.warn("Unknown file " + filename + ". No initial values set.")
      raw_input("Confirm this message to continue:")

  def setStateFilter(self, extractor):
    """
    Set the state filter here, which returns the state representation for learning.
    The default one is an identity function.
    """
    self.getState = extractor

  def getAction(self, state):
    return QLearningAgent.getAction(self, self.getState(state))

  def update(self, state, action, nextState, reward):
    return QLearningAgent.update(self, self.getState(state), action, self.getState(nextState), reward)

  def final(self, state):
    # TODO print values here?
    return QLearningAgent.final(self, self.getState(state))

class ApproximateQAgent(QLearningAgent):
  """
     ApproximateQLearningAgent
     
     You should only have to overwrite getQValue
     and update.  All other QLearningAgent functions
     should work as is.
  """
  def __init__(self, extractor = IdentityExtractor(), **args):
    self.featExtractor = extractor
    QLearningAgent.__init__(self, **args)

    # You might want to initialize weights here.
    "*** YOUR CODE HERE ***"
    self.weights = util.Counter()
    # parameter for BTLS
    self.beta = 0.5

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


class ApproximateVAgent(ApproximateQAgent):
  """
  Approximate the states instead.
  self.weights[action] keeps a directory of weightes for that state, action pair.
  """
  def setWeights(self, filename):
    """
      Set initial weights, if we have it
    """
    if os.path.isfile(filename):
      import pickle
      self.weights = pickle.load(open(filename))
    else:
      import warnings
      warnings.warn("Unknown file " + filename + ". No initial weight set.")
      raw_input("Confirm this message to continue:")

  def getQValue(self, state, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    "*** YOUR CODE HERE ***"
    self.checkAction(action)

    q = 0.0
    feats = self.featExtractor.getStateFeatures(state)
    if feats != None:
      for feature, value in self.featExtractor.getStateFeatures(state).items():
        q += self.weights[action][feature] * value
      return q
    else:
      return 0
    
  def update(self, state, action, nextState, reward):
    """
       Should update your weights based on transition.

       Considering using BTLS.
    """
    self.checkAction(action)
    feats = self.featExtractor.getStateFeatures(state)

    correction = (reward + self.gamma * self.getValue(nextState)) - self.getQValue(state, action)
    # note f(w) = 0.5 * correction ** 2

    t = 1
    thres = 0.00001
    # back tracking line search
    while True:
      for feature, value in feats.items():
        self.weights[action][feature] += t * correction * value
      newQValue = self.getQValue(state, action)
      # revert
      for feature, value in feats.items():
        self.weights[action][feature] -= t * correction * value

      # f(w + t * delta)
      fStep = 0.5 * ((reward + self.gamma * self.getValue(nextState)) - newQValue) ** 2

      # f(w) + alpha * t * Df (w) * delta
      fApprox = 0.5 * correction ** 2 - self.alpha * t * (correction * value) ** 2

      if fStep > fApprox and t > thres:
        t *= self.beta
      else:
        break

    if t < thres:
      # nothing to do, already converged
      return

    if feats != None:
      for feature, value in feats.items():
        self.weights[action][feature] += self.alpha * t * correction * value

  def checkAction(self, action):
    """
    Make sure such action is already in the list.
    If not, create a Counter object for that action.
    """
    if not action in self.weights:
      self.weights[action] = util.Counter()
