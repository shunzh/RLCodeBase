from machineConfig import MachineConfiguration
import JQTPAgent

def main():
  cmp = MachineConfiguration(3, 3, rewardFunc, responseFunc)
  agent = JQTPAgent(cmp, rewardSet)