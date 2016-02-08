from machineConfig import MachineConfiguration
from QTPAgent import AlternatingQTPAgent, JointQTPAgent
import random
import util
import config
import sys
from CMPExp import Experiment
from cmp import QueryType

# reward for incomplete configurations
cost = 0
# discount factor
gamma = 0.8
# the time step that the agent receives the response
responseTime = 2

numMachines = 3
numConfigs = 2

# ask the operator: if this state is selected, what would you do?
queries = [[0] * numMachines for _ in xrange(numMachines)]
for _ in xrange(numMachines): queries[_][_] = 'S'
queries = [tuple(query) for query in queries]

config.VERBOSE = True

def main():
  factoredRewards = []

  """
  rewardNum = 10
  # set the random seed here so the experiments are reproducible
  # read seed from argument
  randomTable = {(idx, i, j): random.random() * 3 for idx in xrange(rewardNum)\
                                                  for i in xrange(numMachines)\
                                                  for j in xrange(numConfigs)}
  
  """
  # test domains
  rewardNum = 2
  randomTable = util.Counter()
  randomTable[(0, 0, 0)] = 5
  randomTable[(0, 1, 0)] = 1
  randomTable[(0, 1, 1)] = -1

  randomTable[(1, 0, 0)] = 5
  randomTable[(1, 1, 0)] = -1
  randomTable[(1, 1, 1)] = 1

  for idx in xrange(rewardNum):
    factoredRewards.append(lambda i, j, idx=idx: randomTable[(idx, i, j-1)])
  rewardSet = map(lambda rf: rewardFuncGen(rf, numMachines), factoredRewards)
  
  # print the true reward
  print [(i, j+1, randomTable[0, i, j]) for i in xrange(numMachines) for j in xrange(numConfigs)]

  rewardNum = len(rewardSet)
  initialPhi = [1.0 / rewardNum] * rewardNum

  Agent = JointQTPAgent
  #Agent = AlternatingQTPAgent

  cmp = MachineConfiguration(numMachines, numConfigs, rewardSet[0], queries, gamma, responseTime)
  agent = Agent(cmp, rewardSet, initialPhi, QueryType.POLICY, gamma)
 
  ret, qValue, timeElapsed = Experiment(cmp, agent, gamma, rewardSet, QueryType.POLICY)
  print ret
  print qValue

  if config.SAVE_TO_FILE:
    rFile = open('results', 'a')
    rFile.write(str(ret) + '\n')
    rFile.close()

    bFile = open('beliefs', 'a')
    bFile.write(str(qValue) + '\n')
    bFile.close()

    tFile = open('time', 'a')
    tFile.write(str(timeElapsed) + '\n')
    tFile.close()

def rewardFuncGen(factorRewardFunc, size):
  def func(s):
    if not 0 in s and not 'S' in s:
      #return cost + gamma * sum([factorRewardFunc(i, s[i]) for i in range(size)]) # rob's way
      return sum([factorRewardFunc(i, s[i]) for i in range(size)])
    else:
      return 0
  return func

if __name__ == '__main__':
  main()
