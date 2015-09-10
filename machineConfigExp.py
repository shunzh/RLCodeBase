from machineConfig import MachineConfiguration
from JQTPAgent import JQTPAgent

def main():
  numMachines = 3
  numConfigs = 3

  queries = range(numMachines) # can ask a configuration of a machine
  factoredReward = lambda i, j: j
  rewardSet = map(lambda rf: rewardFuncGen(rf, numMachines), [factoredReward])

  responseFunc = lambda q: max(range(numMachines), key=lambda config: factoredReward(q, config))
  cmp = MachineConfiguration(numMachines, numConfigs, rewardSet[0], responseFunc, queries)
  
  agent = JQTPAgent(cmp, rewardSet, [1])
  
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

