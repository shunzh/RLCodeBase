import config
import time
import copy

def experiment(cmp, agent, gamma, rewardSet, queryType, horizon=float('inf')):
  t = time.time()

  # code for multi-querying
  """
  def computeQ(agent, t):
    q, pi, qValue = agent.learn()
    #print agent.phi, qValue

    if t > 1:
      qSum = 0
      psiProbs = agent.getPossiblePhiAndProbs(q)
      for psi, prob in psiProbs:
        agent.resetPsi(list(psi))
        q = computeQ(agent, t - 1)
        qSum += prob * qValue
      return q
    else:
      return q
  """

  q, qValue = agent.learn()
  timeElapsed = time.time() - t
  
  if qValue == None:
    qValue = agent.getQValue(agent.cmp.state, None, q)

  priorAgent = agent.getFiniteVIAgent(agent.phi, cmp.horizon, cmp.terminalReward, posterior=True)
  priorV = priorAgent.getValue(cmp.state)
  qValue = qValue - priorV
 
  # add query type here
  if config.VERBOSE: print 'q', q
 
  ret = 0

  return ret, qValue, timeElapsed
