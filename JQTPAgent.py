from learningAgents import ReinforcementAgent

class JPQTAgent:
  def __init__(self, gamma, cmp):
    self.gamma = gamma
    self.cmp = cmp
    
    self.n = self.cmp.n
    # init belief on rewards with uniform distribution
    self.phi = [1.0 / self.n] * self.n

  def getValue(self, pi=None, horizon='inf'):


  def getQValue(self, state, policy, query):
    cost = self.cmp.cost(query)

    responseTime = self.cmp.getResponseTime(state)
    vBeforeResponse = self.getValue(state, pi=policy, horizon=responseTime)
    vAfterResponse = self.gamma ** responseTime * 
    
    return cost + vBeforeResponse