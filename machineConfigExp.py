from machineConfig import MachineConfiguration
from QTPAgent import IterativeQTPAgent, JointQTPAgent
import random
from baselineAgents import RandomAgent
import util
import config
import sys

# reward for incomplete configurations
cost = -0.2
# discount factor
gamma = 0.9
# the time step that the agent receives the response
responseTime = 2

def JQTPExp(cmp, agent, rewardSet, queryEnabled=True):
  q, pi, qValue = agent.learn()
 
  # init state
  state = cmp.state
  print 's', state, 'r', cmp.getReward(state)

  # accumulated return
  ret = 0
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
    
    action = pi(state, cmp.timer)
    state, reward = cmp.doAction(action)
    print 's', state, 'r', reward

    cmp.timeElapse()

    ret += reward * gamma ** cmp.timer
  
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
  # set the random seed here so the experiments are reproducible
  # read seed from argument
  random.seed(sys.argv[1])
  randomTable = {(idx, i, j): random.random() * 3 for idx in xrange(rewardNum)\
                                                  for i in xrange(numMachines)\
                                                  for j in xrange(numConfigs)}
  """
  rewardNum = 2
  randomTable = util.Counter()
  randomTable[(0, 0, 0)] = 5
  randomTable[(1, 0, 0)] = 5
  
  randomTable[(0, 1, 0)] = 1
  randomTable[(1, 1, 1)] = 1

  randomTable[(0, 2, 0)] = 1
  randomTable[(1, 2, 1)] = 1
  
  for idx in xrange(rewardNum):
    factoredRewards.append(lambda i, j, idx=idx: randomTable[(idx, i, j-1)])
  rewardSet = map(lambda rf: rewardFuncGen(rf, numMachines), factoredRewards)
  
  queries = [('S', 0, 0), (0, 'S', 0), (0, 0, 'S')]

  # print the true reward
  print [(i, j+1, randomTable[0, i, j]) for i in xrange(numMachines) for j in xrange(numConfigs)]

  queryEnabled = True
  queryIgnored = False

  rewardNum = len(rewardSet)
  initialPhi = [1.0 / rewardNum] * rewardNum

  Agent = JointQTPAgent
  #Agent = IterativeQTPAgent

  cmp = MachineConfiguration(numMachines, numConfigs, rewardSet[0], queries, gamma, responseTime)
  agent = Agent(cmp, rewardSet, initialPhi, gamma=gamma,\
                queryEnabled=queryEnabled, queryIgnored=queryIgnored)
 
  ret, qValue = JQTPExp(cmp, agent, rewardSet, queryEnabled)
  print ret
  print qValue

  if config.SAVE_TO_FILE:
    rFile = open(Agent.__name__ + 'results', 'a')
    rFile.write(str(ret) + '\n')
    rFile.close()

    bFile = open(Agent.__name__ + 'beliefs', 'a')
    bFile.write(str(qValue) + '\n')
    bFile.close()

def rewardFuncGen(factorRewardFunc, size):
  def func(s):
    if not 0 in s and not 'S' in s:
      # Rob may have discounted this in the final step
      return sum([factorRewardFunc(i, s[i]) for i in range(size)])
    else:
      return cost
  return func

if __name__ == '__main__':
  main()
