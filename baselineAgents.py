from valueIterationAgents import ValueIterationAgent
import random

class RandomAgent(ValueIterationAgent):
  """
  Choose a random action as policy
  """
  def __init__(self, mdp):
    ValueIterationAgent.__init__(self, mdp)

  def learn(self):
    return lambda s: random.choice(self.mdp.getPossibleActions(s))
