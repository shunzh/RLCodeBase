from QTPAgent import IterativeQTPAgent, JointQTPAgent
from CMPExp import Experiment
import util
from robotNavigation import RobotNavigation

width = 3
height = 3

# discount factor
gamma = 0.9
# the time step that the agent receives the response
responseTime = 8

queries = [(3, 2)]

def main():
  rewards0 = util.Counter()
  rewards0[(2, 0)] = 10
  rewards0[(2, 2)] = -10
  
  rewards1 = util.Counter()
  rewards1[(2, 0)] = -10
  rewards1[(2, 2)] = 10

  rewardSet = [rewardGen(rewards0), rewardGen(rewards1)]
  rewardNum = len(rewardSet)
  initialPhi = [1.0 / rewardNum] * rewardNum

  Agent = JointQTPAgent
  cmp = RobotNavigation(queries, rewardSet[1], gamma, responseTime, width, height)
  agent = Agent(cmp, rewardSet, initialPhi, gamma=gamma)
 
  ret, qValue = Experiment(cmp, agent, gamma, rewardSet, horizon=10)
  print ret
  print qValue

def rewardGen(rewards): 
  def rewardFunc(s):
    if s in rewards.keys():
      return rewards[s]
    else:
      return 0
  return rewardFunc

if __name__ == '__main__':
  main()
