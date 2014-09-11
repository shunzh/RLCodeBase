import ModularAgents from modularAgents

class InverseModularRL:
  """
    Inverse Reinforcement Learning. From a trained modular agent, find the weights given
    - Q tables of modules
    - Policy function
  """

  def __init__(self, agent, mdp, qFuncs):
    """
      Args:
        agent: the modular agent object
        mdp: the hybird environment
        qFuncs: 
    """
    self.agent = agent
    self.mdp = mdp
    self.qFuncs = qFuncs

    self.maxSweepTimes = 10

  def findWeights(self):
    """
      Find the approporiate weight for each module, by walking through the policy
    """
    states = self.mdp.getStates()

    for _ in xrange(self.maxSweepTimes):
      # Walk through each state
      for state in states:
        optAction = self.agent.getPolicy(state)
        # Update the weights for each module accordingly.
        for qFunc in self.qFuncs:
          qValues = 
          maxQValue = 
