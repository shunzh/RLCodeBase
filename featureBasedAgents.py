from QTPAgent import GreedyConstructionPiAgent

class FeatureBasedPolicyQueryAgent(GreedyConstructionPiAgent):
  def __init__(self, cmp, rewardSet, initialPhi, queryType, feat, featLength, gamma):
    self.feat = feat
    self.featLength = featLength

    # step size for local search
    self.epsilon = 0.01
    GreedyConstructionPiAgent.__init__(self, cmp, rewardSet, initialPhi, queryType, gamma)

  def findNextPolicy(self, S, A, R, T, s0, psi, maxV, q):
    """
    We exploit the fact that the reward candidates are linearly parameterizable.
    We find all the verticies formed by previous policies, and optimize EUS locally.
    
    Note that R is a set of parameters.
    """