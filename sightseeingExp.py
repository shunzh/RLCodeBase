from QTPAgent import IterativeQTPAgent, JointQTPAgent
from CMPExp import Experiment
import util
from sightseeing import Sightseeing

width = 10
height = 10

# discount factor
gamma = 0.9
# the time step that the agent receives the response
responseTime = 5

queries = [(1, 1, 0), (5, 9, 0), (9, 5, 0)]

def main():
  rewards = []

  for query in queries:
    x, y, status = query
    reward = util.Counter()
    reward[(x, y)] = 1
    rewards.append(reward)

  rewardSet = [rewardGen(reward) for reward in rewards]
  rewardNum = len(rewardSet)
  initialPhi = [1.0 / rewardNum] * rewardNum

  Agent = JointQTPAgent
  #Agent = IterativeQTPAgent
  cmp = Sightseeing(queries, rewardSet[0], gamma, responseTime, width, height)
  agent = Agent(cmp, rewardSet, initialPhi, gamma=gamma)
 
  ret, qValue = Experiment(cmp, agent, gamma, rewardSet)
  print ret
  print qValue

def rewardGen(rewards): 
  def rewardFunc(s):
    x, y, status = s
    if status == 1:
      if (x, y) in rewards.keys():
        return rewards[(x, y)]
      else:
        return -1
    else:
      return 0
  return rewardFunc

if __name__ == '__main__':
  main()
