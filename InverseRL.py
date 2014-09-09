import ModularAgents from modularAgents

class InverseRL:
  """
    Inverse Reinforcement Learning. From a trained modular agent, find the weights given
    - Q tables of modules
    - Policy function
  """

  def __init__(self, agent, qFuncs):
    """
      Args:
        agent: the modular agent object
        qFuncs: 
    """
    self.agent = agent
    self.qFuncs = qFuncs
