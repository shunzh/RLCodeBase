from QTPAgent import IterativeQTPAgent, JointQTPAgent
from windyCorridor import WindyCorridor
from CMPExp import Experiment
import util

# discount factor
gamma = 0.9
# the time step that the agent receives the response
responseTime = 5

interLength = 3
interNum = 3
circular = True

# at this intersection, what would you do?
queries = [(interId, interLength - 1) for interId in xrange(interNum)]

def main():
  rewards = []

  reward = util.Counter()
  reward[(0, 'L')] = 10
  reward[(0, 'R')] = -10
  reward[(1, 'L')] = 1
  reward[(1, 'R')] = -1
  rewards.append(reward)

  reward = util.Counter()
  reward[(0, 'L')] = 10
  reward[(0, 'R')] = -10
  reward[(1, 'L')] = -1
  reward[(1, 'R')] = 1
  rewards.append(reward)

  reward = util.Counter()
  reward[(0, 'L')] = -10
  reward[(0, 'R')] = 10
  reward[(1, 'L')] = 1
  reward[(1, 'R')] = -1
  rewards.append(reward)

  reward = util.Counter()
  reward[(0, 'L')] = -10
  reward[(0, 'R')] = 10
  reward[(1, 'L')] = -1
  reward[(1, 'R')] = 1
  rewards.append(reward)

  rewardSet = [rewardGen(reward) for reward in rewards]

  rewardNum = len(rewardSet)
  initialPhi = [1.0 / rewardNum] * rewardNum

  Agent = JointQTPAgent
  cmp = WindyCorridor(queries, rewardSet[0], gamma, responseTime, interLength, interNum, circular)
  agent = Agent(cmp, rewardSet, initialPhi, gamma=gamma)
 
  ret, qValue = Experiment(cmp, agent, gamma, rewardSet, horizon=interLength * interNum * 2)
  print ret
  print qValue

def rewardGen(rewards): 
  def rewardFunc(s):
    return rewards[s]
  return rewardFunc

if __name__ == '__main__':
  main()
