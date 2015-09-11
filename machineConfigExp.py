from machineConfig import MachineConfiguration
from JQTPAgent import JQTPAgent

def main():
  numMachines = 3
  numConfigs = 3

  factoredReward = lambda i, j: j
  alterFactoredReward = lambda i, j: 3 - j
  rewardSet = map(lambda rf: rewardFuncGen(rf, numMachines), [factoredReward, alterFactoredReward])

  # can ask a configuration of a machine
  queries = [(0, 1, 1), (1, 0, 1), (1, 1, 0)]
  cmp = MachineConfiguration(numMachines, numConfigs, rewardSet[0], queries)
  
  initialPhi = [.5, .5]
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

