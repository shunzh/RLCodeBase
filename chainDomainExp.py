from chainDomain import ChainDomain
from JQTPAgent import JQTPAgent

def main():
  def rewardFunc(state):
    if state == 0: return 10
    else: return 0

  def alterRewardFunc(state):
    if state == 4: return 1
    else: return 0

  gamma = .9
  queries = [2]

  cmp = ChainDomain(queries, alterRewardFunc)
  agent = JQTPAgent(cmp, [rewardFunc, alterRewardFunc], [.5, .5], gamma=gamma)
  
  q, pi = agent.learn()
  state = cmp.state
  print 's', state
  ret = 0

  while True:
    if cmp.isTerminal(state):
      break
    
    # query the model in the first time step
    if cmp.timer == 0:
      cmp.query(q)
    
    # see whether there is any response
    response = cmp.responseCallback()
    if response != None:
      # update policy
      pi = agent.respond(q, response)
    
    action = pi(state)
    state, reward = cmp.doAction(action)
    print 's', state, 'r', reward
    ret += reward * gamma ** cmp.timer
  
  print ret
 
if __name__ == '__main__':
  main()