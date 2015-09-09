from chainDomain import ChainDomain
from JQTPAgent import JPQTAgent

def main():
  def rewardFunc(state):
    if state == 0: return 10
    elif state == 4: return 0

  def alterRewardFunc(state):
    if state == 0: return -1
    elif state == 4: return 1

  cmp = ChainDomain(rewardFunc)
  agent = JPQTAgent(cmp, [rewardFunc, alterRewardFunc], [.5, .5])
  
  agent.learn()