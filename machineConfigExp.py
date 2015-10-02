from machineConfig import MachineConfiguration
from QTPAgent import IterativeQTPAgent, JointQTPAgent
import random
import util
import config
import sys
from CMPExp import Experiment

# reward for incomplete configurations
cost = -0.2
# discount factor
gamma = 0.9
# the time step that the agent receives the response
responseTime = 2

numMachines = 5
numConfigs = 3

# ask the operator: if this state is selected, what would you do?
queries = [[0] * numMachines for _ in xrange(numMachines)]
for _ in xrange(numMachines): queries[_][_] = 'S'
queries = [tuple(query) for query in queries]
#queries = [(0, 0, 0)]

# for OQPP
queryIgnored = False

def main():
  factoredRewards = []

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
  randomTable[(0, 1, 0)] = 0
  randomTable[(0, 1, 1)] = 2
  randomTable[(0, 2, 0)] = 3
  randomTable[(0, 2, 1)] = 5
  randomTable[(0, 2, 2)] = 5

  randomTable[(1, 0, 0)] = 5
  randomTable[(1, 1, 0)] = 2
  randomTable[(1, 1, 1)] = 1
  randomTable[(1, 2, 0)] = 5
  randomTable[(1, 2, 1)] = 4
  randomTable[(1, 2, 2)] = 3
  """

  for idx in xrange(rewardNum):
    factoredRewards.append(lambda i, j, idx=idx: randomTable[(idx, i, j-1)])
  rewardSet = map(lambda rf: rewardFuncGen(rf, numMachines), factoredRewards)
  
  # print the true reward
  print [(i, j+1, randomTable[0, i, j]) for i in xrange(numMachines) for j in xrange(numConfigs)]

  rewardNum = len(rewardSet)
  initialPhi = [1.0 / rewardNum] * rewardNum

  Agent = JointQTPAgent
  #Agent = IterativeQTPAgent

  cmp = MachineConfiguration(numMachines, numConfigs, rewardSet[0], queries, gamma, responseTime)
  agent = Agent(cmp, rewardSet, initialPhi, gamma=gamma,\
                queryIgnored=queryIgnored)
 
  ret, qValue = Experiment(cmp, agent, gamma, rewardSet)
  print ret
  print qValue

  if config.SAVE_TO_FILE:
    rFile = open('results' + sys.argv[1], 'a')
    rFile.write(str(ret) + '\n')
    rFile.close()

    bFile = open('beliefs' + sys.argv[1], 'a')
    bFile.write(str(qValue) + '\n')
    bFile.close()

def rewardFuncGen(factorRewardFunc, size):
  def func(s):
    if not 0 in s and not 'S' in s:
      #return cost + gamma * sum([factorRewardFunc(i, s[i]) for i in range(size)]) # rob's way
      return sum([factorRewardFunc(i, s[i]) for i in range(size)])
    else:
      return cost
  return func

if __name__ == '__main__':
  main()
