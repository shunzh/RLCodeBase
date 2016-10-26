import config
import time

def experiment(cmp, agent, gamma, rewardSet, queryType, times=1, horizon=float('inf')):
  t = time.time()

  for _ in xrange(times):
    q, pi, qValue = agent.learn()

    cmp.query(q)
    res = cmp.responseCallback(agent)

    # update belief
    agent.phi = list(agent.responseToPhi[(tuple(q), res)])
    print agent.phi

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