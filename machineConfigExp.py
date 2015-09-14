from machineConfig import MachineConfiguration
from JQTPAgent import JQTPAgent
import random
import copy
import util

cost = -0.2
gamma = 0.9

def main():
  factoredRewards = []

  numMachines = 3
  numConfigs = 3

  rewardNum = 10
  randomTable = {(idx, i, j): random.random() * 3 for idx in xrange(rewardNum)\
                                                  for i in xrange(numMachines)\
                                                  for j in xrange(numConfigs)}
  """
  randomTable = {(idx, i, j): j+1 for idx in xrange(rewardNum)\
                                  for i in xrange(numMachines)\
                                  for j in xrange(numConfigs)}
  """
  queries = [(0, 1, 1), (1, 0, 1), (1, 1, 0)]

  # FIXME think of a good way to do this!!
  for idx in xrange(rewardNum):
    factoredRewards.append(lambda i, j, idx=idx: randomTable[(idx, i, j-1)])

  rewardSet = map(lambda rf: rewardFuncGen(rf, numMachines), factoredRewards)
  
  # can ask a configuration of a machine
  cmp = MachineConfiguration(numMachines, numConfigs, rewardSet[0], queries)
  
  print [(i, j+1, randomTable[0, i, j]) for i in xrange(numMachines) for j in xrange(numConfigs)]

  initialPhi = [1.0 / rewardNum] * rewardNum
  agent = JQTPAgent(cmp, rewardSet, initialPhi, gamma=gamma ** 2)
  
  q, pi, qValue = agent.learn()
  ret = 0
  
  # init state
  state = cmp.state
  print 's', state
  # accumulated return
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
      print 'o', response
      pi = agent.respond(q, response)
    
    action = pi(state)
    state, reward = cmp.doAction(action)
    print 's', state, 'r', reward

    ret += reward * gamma ** ((cmp.timer - 1) * 2)
  
  print 'return', ret
  print 'qvalue', qValue

  """
  rFile = open('results', 'a')
  rFile.write(str(ret) + '\n')
  rFile.close()

  bFile = open('beliefs', 'a')
  bFile.write(str(qValue) + '\n')
  bFile.close()
  """

def rewardFuncGen(factorRewardFunc, size):
  def func(s):
    if not 0 in s:
      return cost + gamma * sum([factorRewardFunc(i, s[i]) for i in range(size)])
    else:
      return cost * (1 + gamma)
  return func

if __name__ == '__main__':
  main()
