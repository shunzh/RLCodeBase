# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util, sys

from learningAgents import ValueEstimationAgent
import random
INF = sys.maxint

class ValueIterationAgent(ValueEstimationAgent):
  """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """
  def __init__(self, mdp, discount, iterations = INF, initValues = util.Counter()):
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
    # allValues: t, state -> value
    self.allValues = [initValues]
  
  def learn(self):
    for t in xrange(self.iterations):
      # note that values are updated off-place
      values = util.Counter()
      for state in self.mdp.getStates():
        if self.mdp.isTerminal(state):
          # the state that represents end of a episode
          # the value is 0 on this state
          values[state] = 0
        else:
          values[state] = max([self.getQValue(state, action, t)\
                               for action in self.mdp.getPossibleActions(state)])

      self.allValues.append(values)

      #print util.getValueDistance(values, self.allValues[-2])
      # Can stop for infinite horizon and converged
      if self.iterations == INF and util.getValueDistance(values, self.allValues[-2]) < .01:
        break

    self.allValues.reverse()
    
  def getValue(self, state, t=0):
    """
      Return the value of the state
      By default, look at the value in the last timestep
    """
    return self.allValues[t][state]

  def getQValue(self, state, action, t=0):
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
      q += prob * (self.mdp.getReward(state, action) + self.discount * self.getValue(nextState, t))
    return q
    
  def getPolicy(self, state, t=0):
    """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
    return random.choice(self.getPolicies(state, t))
    #return self.getPolicies(state)[0] # enable this to fix randomness
  
  def getPolicies(self, state, t=0):
    """
    Return the actions which share the max q
    """
    if self.mdp.isTerminal(state):
      return []
    else:
      maxValue = -INF
      actions = self.mdp.getPossibleActions(state)
      for action in actions:
        q = self.getQValue(state, action, t+1)
        if q > maxValue:
          maxValue = q
      return filter(lambda act: self.getQValue(state, act, t+1) == maxValue, actions)
