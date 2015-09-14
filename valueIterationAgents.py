# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util, sys

from learningAgents import ValueEstimationAgent
INF = sys.maxint

class ValueIterationAgent(ValueEstimationAgent):
  """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """
  def __init__(self, mdp, discount = 0.9, iterations = INF, initValues = util.Counter()):
    """
      Your value iteration agent should take an mdp on
      construction, run the indicated number of iterations
      and then act according to the resulting policy.
    
      Some useful mdp methods you will use:
          mdp.getStates()
          mdp.getPossibleActions(state)
          mdp.getTransitionStatesAndProbs(state, action)
          mdp.getReward(state, action, nextState)
    """
    self.mdp = mdp
    self.discount = discount
    self.iterations = iterations
    self.values = initValues
     
    for _ in xrange(iterations):
      # note that values are updated off-place
      values = util.Counter()
      for state in mdp.getStates():
        if mdp.isTerminal(state): 
          values[state] = self.mdp.getReward(state)
        else: 
          maxValue = -INF
          for action in mdp.getPossibleActions(state):
            maxValue = max(maxValue, self.getQValue(state, action))
          values[state] = maxValue

      if util.getVectorDistance(values.values(), self.values.values()) < .01:
        # converged
        break
      self.values = values
    
  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """
    return self.values[state]

  def getQValue(self, state, action):
    """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
    "*** YOUR CODE HERE ***"
    q = 0
    for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
      q += prob * (self.mdp.getReward(state) + self.discount * self.getValue(nextState))
    return q
    
  def getPolicy(self, state):
    """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
    "*** YOUR CODE HERE ***"
    if self.mdp.isTerminal(state):
      # no action taken on terminal state
      return None
    else:
      maxValue = -INF
      for action in self.mdp.getPossibleActions(state):
        q = self.getQValue(state, action)
        if q > maxValue:
          maxValue = q
          bestAction = action
      return bestAction
  
  def getPolicies(self, state):
    """
    Return the actions which share the max q
    """
    if self.mdp.isTerminal(state):
      return None
    else:
      maxValue = -INF
      actions = self.mdp.getPossibleActions(state)
      for action in actions:
        q = self.getQValue(state, action)
        if q > maxValue:
          maxValue = q
      return filter(lambda act: self.getQValue(state, act) == maxValue, actions)

  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)
  
