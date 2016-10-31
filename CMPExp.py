import config
import time
import copy

def experiment(cmp, agent, gamma, rewardSet, queryType, horizon=float('inf')):
  t = time.time()

  def computeQ(agent, t):
    q, pi, qValue = agent.learn()
    #print agent.phi, qValue

    if t > 1:
      qSum = 0
      psiProbs = agent.getPossiblePhiAndProbs(q)
      for psi, prob in psiProbs:
        agent.resetPsi(list(psi))
        q, pi, qValue = computeQ(agent, t - 1)
        qSum += prob * qValue
      return q, pi, qSum
    else:
      return q, pi, qValue

  q, pi, qValue = computeQ(agent, config.NUMBER_OF_QUERIES)
  timeElapsed = time.time() - t
  
  # add query type here
  if config.VERBOSE: print 'q', q
 
  ret = 0

  """
  # pi may not be computed, so disabled simulation
  # init state
  state = cmp.state
  if config.VERBOSE: print 's', state

  while True:
    if cmp.isTerminal(state) or cmp.timer >= horizon:
      break
    
    # query the model in the first time step
    if cmp.timer == 0:
      cmp.query(q)
  
    # see whether there is any response
    response = cmp.responseCallback()
    if response != None:
      if config.VERBOSE: print 'o', response
      # update policy
      pi = agent.respond(q, response)
    
    action = pi(state, cmp.timer)
    state, reward = cmp.doAction(action)
    cmp.timeElapse()
    if config.VERBOSE: print cmp.timer, 'a', action, 's', state, 'r', reward
    if config.PRINT == 'states': print state[0], state[1]

    ret += reward * gamma ** cmp.timer
  """
  
  return ret, qValue, timeElapsed