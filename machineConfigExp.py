from machineConfig import MachineConfiguration
from JQTPAgent import JQTPAgent
import random
import copy
import util

gamma = 0.81

def main():
  numMachines = 3
  numConfigs = 3

  factoredRewards = []
  rewardNum = 10
  randomTable = {(idx, i, j): random.random() * 3 for idx in xrange(rewardNum)\
                                                  for i in xrange(numMachines)\
                                                  for j in xrange(numConfigs)}
  """
  rewardNum = 2
  randomTable = util.Counter()
  randomTable[0, 2, 1] = 5
  randomTable[0, 1, 2] = 1
  randomTable[0, 0, 1] = 1
  
  randomTable[1, 2, 1] = 5
  randomTable[1, 1, 1] = 1
  randomTable[1, 0, 2] = 1
  """

  # FIXME think of a good way to do this!!
  for idx in xrange(rewardNum):
    factoredRewards.append(lambda i, j, idx=idx: randomTable[(idx, i, j-1)])

  rewardSet = map(lambda rf: rewardFuncGen(rf, numMachines), factoredRewards)
  
  # can ask a configuration of a machine
  queries = [(0, 1, 1), (1, 0, 1), (1, 1, 0)]
  cmp = MachineConfiguration(numMachines, numConfigs, rewardSet[0], queries)
  
  initialPhi = [1.0 / rewardNum] * rewardNum
  agent = JQTPAgent(cmp, rewardSet, initialPhi, gamma=gamma)
  
  q, pi = agent.learn()
  
  # init state
  state = cmp.state
  print 's', state
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
    if response != None:
      # update policy
      pi = agent.respond(q, response)
    
    action = pi(state)
    state, reward = cmp.doAction(action)
    print 's', state, 'r', reward
    ret += reward * gamma ** cmp.timer
  
  print ret

def rewardFuncGen(factorRewardFunc, size):
  def func(s):
    if not 0 in s:
      return -0.2 + gamma * sum([factorRewardFunc(i, s[i]) for i in range(size)])
    else:
      return -0.2 * (1 + gamma)
  return func

if __name__ == '__main__':
  main()
