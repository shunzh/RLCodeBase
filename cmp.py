from mdp import MarkovDecisionProcess

class ControlledMarkovProcess(MarkovDecisionProcess):
  def __init__(self, queries, responseFunc):
    MarkovDecisionProcess.__init__(self)
    
    self.outsandingQueries = None
    # possible queries
    self.queries = queries
    # a response function that answers the query
    self.responseFunc = responseFunc

  def query(self, q):
    """
    Ask the policy of this state
    """
    self.outsandingQueries = (q, self.timer + self.getResponseTime)
  
  def cost(self, q):
    abstract
  
  def responseCallback(self):
    if self.timer >= self.outsandingQueries[1]:
      # issues a response
      res = self.response(self.outsandingQueries[0])
    
    self.outsandingQueries = None
    return res