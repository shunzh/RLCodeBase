from machineConfig import MachineConfiguration
from JQTPAgent import JQTPAgent
import random
import copy

def main():
  numMachines = 3
  numConfigs = 3
  rewardNum = 10
  gamma = 0.81

  factoredRewards = []
  randomTable = {(idx, i, j): random.random() * 3 for idx in xrange(rewardNum)\
                                                  for i in xrange(numMachines)\
                                                  for j in xrange(numConfigs)}

  # FIXME think of a good way to do this!!
  factoredRewards.append(lambda i, j: randomTable[(0, i, j-1)])
  factoredRewards.append(lambda i, j: randomTable[(1, i, j-1)])
  factoredRewards.append(lambda i, j: randomTable[(2, i, j-1)])
  factoredRewards.append(lambda i, j: randomTable[(3, i, j-1)])
  factoredRewards.append(lambda i, j: randomTable[(4, i, j-1)])
  factoredRewards.append(lambda i, j: randomTable[(5, i, j-1)])
  factoredRewards.append(lambda i, j: randomTable[(6, i, j-1)])
  factoredRewards.append(lambda i, j: randomTable[(7, i, j-1)])
  factoredRewards.append(lambda i, j: randomTable[(8, i, j-1)])
  factoredRewards.append(lambda i, j: randomTable[(9, i, j-1)])

  rewardSet = map(lambda rf: rewardFuncGen(rf, numMachines), factoredRewards)
  
  # can ask a configuration of a machine
  queries = [(0, 1, 1), (1, 0, 1), (1, 1, 0)]
  cmp = MachineConfiguration(numMachines, numConfigs, rewardSet[0], queries)
  
  initialPhi = [1.0 / rewardNum] * rewardNum
  agent = JQTPAgent(cmp, rewardSet, initialPhi, gamma=gamma)
  
  q, pi = agent.learn()
  
  # init state
  state = cmp.state
  # accumulated return
  ret = 0
  while True:
    if cmp.isTerminal(state):
      break
    
    # query the model in the first time step
    if cmp.timer == 0:
      cmp.query(q)
    
    # see whether there is any response
    response = cmp.responseCallback()
    #if response != None:
      # update policy
      #pi = agent.respond(q, response)
    
    action = pi(state)
    state, reward = cmp.doAction(action)
    ret += reward * gamma ** cmp.timer
  
  print ret

def rewardFuncGen(factorRewardFunc, size):
  def func(s):
    if not 0 in s:
      return sum([factorRewardFunc(i, s[i]) for i in range(size)])
    else:
      return -0.2
  return func

if __name__ == '__main__':
  main()
