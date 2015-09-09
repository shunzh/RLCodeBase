from chainDomain import ChainDomain
from JQTPAgent import JPQTAgent

def main():
  def rewardFunc(state):
    if state == 0: return 10
    else: return 0

  def alterRewardFunc(state):
    if state == 4: return 1
    else: return 0

  queries = [2]

  cmp = ChainDomain(rewardFunc, queries)
  agent = JPQTAgent(cmp, [rewardFunc, alterRewardFunc], [.5, .5])
  
  agent.learn()

if __name__ == '__main__':
  main()