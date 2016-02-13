from QTPAgent import QTPAgent

class QueryActionAgent(QTPAgent):
  """
  Allowed to ask one query in the whole process, response returned immediately.
  """
  def __init__(self, cmp, rewardSet, initialPhi, queryType, gamma):
    QTPAgent.__init__(self, cmp, rewardSet, initialPhi, queryType, gamma)
    
  def augmentCMP(self):
    """
    This copies this CMP |{q_i}| times.
    For each CMP, it assigns the reward using possible posterior belief.
    """
    cmp = self.cmp
    
    for query in self.cmp.queries:
      possiblePhis = self.getPossiblePhiAndProbs(query)
      for fPhi, fPhiProb in possiblePhis: