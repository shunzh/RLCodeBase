import config
def Experiment(cmp, agent, gamma, rewardSet):
  q, pi, qValue = agent.learn()
 
  # init state
  state = cmp.state
  if config.VERBOSE: print 's', state, 'r', cmp.getReward(state)

  # accumulated return
  ret = cmp.getReward(state)
  while True:
    if cmp.isTerminal(state):
      break
    
    # query the model in the first time step
    if cmp.timer == 0:
      if config.VERBOSE: print 'q', q
      cmp.query(q)
  
    # see whether there is any response
    response = cmp.responseCallback()
    if response != None:
      if config.VERBOSE: print 'o', response
      # update policy
      pi = agent.respond(q, response)
    
    action = pi(state, cmp.timer)
    state, reward = cmp.doAction(action)
    if config.VERBOSE: print 's', state, 'r', reward

    cmp.timeElapse()
    ret += reward * gamma ** cmp.timer
  
  return ret, qValue