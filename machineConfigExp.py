from machineConfig import MachineConfiguration
from JQTPAgent import JQTPAgent
import random

def main():
  numMachines = 3
  numConfigs = 3
  rewardNum = 10

  factoredRewards = []
  randomTable = [random.random() * 3 for _ in xrange(numMachines * (numConfigs+1) * rewardNum)]
  for idx in range(rewardNum):
    rewardFunc = lambda i, j: randomTable[idx * numMachines * numConfigs + i * numMachines + j - 1]
    factoredRewards.append(rewardFunc)
  rewardSet = map(lambda rf: rewardFuncGen(rf, numMachines), factoredRewards)

  # can ask a configuration of a machine
  queries = [(0, 1, 1), (1, 0, 1), (1, 1, 0)]
  cmp = MachineConfiguration(numMachines, numConfigs, rewardSet[0], queries)
  
  initialPhi = [1.0 / rewardNum] * rewardNum
  agent = JQTPAgent(cmp, rewardSet, initialPhi)
  
  agent.learn()

def rewardFuncGen(factorRewardFunc, size):
  def func(s):
    if not 0 in s:
      return sum([factorRewardFunc(i, s[i]) for i in range(size)])
    else:
      return -0.2
  return func

if __name__ == '__main__':
  main()

