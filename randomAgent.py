import random

class RandomAgent:
  def __init__(self, mdp):
    self.mdp = mdp

  def getAction(self, state):
    return self.getPolicy(state)

  def getValue(self, state):
    return 0.0

  def getQValue(self, state, action):
    return 0.0

  def getPolicy(self, state):
    return random.choice(self.mdp.getPossibleActions(state))

  def update(self, state, action, nextState, reward):
    pass      