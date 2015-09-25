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
circular = False

# at this intersection, what would you do?
queries = [(interId, interLength - 1) for interId in xrange(interNum)]

def main():
  reward0 = util.Counter()
  reward0[(2, 'L')] = 10
  reward0[(0, 'L')] = -10

  reward1 = util.Counter()
  reward1[(2, 'R')] = 10
  reward1[(0, 'R')] = -10

  rewardSet = [rewardGen(reward0), rewardGen(reward1)]

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
