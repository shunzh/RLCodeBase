class InverseRL:
  """
  Inverse Reinforcement Learning algorithm.

  This is implemented as
  Ng, Andrew Y., and Stuart J. Russell.
  "Algorithms for inverse reinforcement learning."
  ICML. 2000.
  """
  def __init__(self, mdp = None):
    self.mdp = mdp

  def solve(self):
    #TODO
    # reward function for training
    def rewardFuncGenerator(alpha):
      def getReward(worldSelf, state, action, nextState):
        # dot(alpha, nextState)
        return None
      
      return getReward