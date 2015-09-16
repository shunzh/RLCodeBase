from machineConfig import MachineConfiguration
from QTPAgent import IterativeQTPAgent, JointQTPAgent
import random
from baselineAgents import RandomAgent
import util

cost = -0.2
gamma = 0.9

def JQTPExp(cmp, agent, rewardSet, queryEnabled=True):
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
    if queryEnabled:
      if cmp.timer == 0:
        print 'q', q
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

    ret += reward * gamma ** (cmp.timer * 2)
    
    cmp.timeElapse()
  
  return ret, qValue

def priorSanityCheck(numConfigs, rewardNum, randomTable):
  meanRewards = [sum([randomTable[idx, 0, j] for idx in range(rewardNum)])\
                                             for j in range(numConfigs)]
  maxMeanConfig = max(range(numConfigs), key=lambda j: meanRewards[j])
  # assume first reward is the true reward
  return randomTable[0, 0, maxMeanConfig]

def main():
  factoredRewards = []

  numMachines = 3
  numConfigs = 3

  """
  rewardNum = 10
  randomTable = {(idx, i, j): random.random() * 3 for idx in xrange(rewardNum)\
                                                  for i in xrange(numMachines)\
                                                  for j in xrange(numConfigs)}
  """
  rewardNum = 2
  randomTable = util.Counter()
  randomTable[(0, 0, 0)] = 5
  randomTable[(1, 0, 0)] = 5
  
  randomTable[(0, 1, 0)] = 2
  randomTable[(0, 1, 1)] = -2
  randomTable[(0, 2, 0)] = 2
  randomTable[(0, 2, 1)] = -2

  randomTable[(1, 1, 0)] = -1
  randomTable[(1, 1, 1)] = 1
  randomTable[(1, 2, 0)] = -1
  randomTable[(1, 2, 1)] = 1
  
  for idx in xrange(rewardNum):
    factoredRewards.append(lambda i, j, idx=idx: randomTable[(idx, i, j-1)])
  rewardSet = map(lambda rf: rewardFuncGen(rf, numMachines), factoredRewards)
  
  queries = [(0, 1, 1), (1, 0, 1), (1, 1, 0)]

  # print the true reward
  print [(i, j+1, randomTable[0, i, j]) for i in xrange(numMachines) for j in xrange(numConfigs)]

  queryEnabled = True
  queryIgnored = False

  rewardNum = len(rewardSet)
  initialPhi = [1.0 / rewardNum] * rewardNum

  #Agent = IterativeQTPAgent
  Agent = JointQTPAgent

  cmp = MachineConfiguration(numMachines, numConfigs, rewardSet[0], queries, responseTime=1)
  agent = Agent(cmp, rewardSet, initialPhi, gamma=gamma ** 2,\
                queryEnabled=queryEnabled, queryIgnored=queryIgnored)
 
  ret, qValue = JQTPExp(cmp, agent, rewardSet, queryEnabled)
  print ret
  print qValue

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
      return cost * (1 + gamma) + gamma ** 2 * (sum([factorRewardFunc(i, s[i]) for i in range(size)]))
    else:
      return cost * (1 + gamma)
  return func

if __name__ == '__main__':
  main()
