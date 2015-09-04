from mdp import MarkovDecisionProcess

class ControlledMarkovProcess(MarkovDecisionProcess):
  def __init__(self, responseFunc):
    MarkovDecisionProcess.__init__(self)
    
    self.outsandingQueries = None
    self.responseFunc = responseFunc

  def query(self, q):
    """
    Ask the policy of this state
    """
    self.outsandingQueries = (q, self.timer + self.getResponseTime)
  
  def responseCallback(self):
    if self.timer >= self.outsandingQueries[1]:
      # issues a response
      res = self.response(self.outsandingQueries[0])
    
    self.outsandingQueries = None
    return res