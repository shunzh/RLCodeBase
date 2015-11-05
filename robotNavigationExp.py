from QTPAgent import IterativeQTPAgent, JointQTPAgent
from CMPExp import Experiment
import util
import sys
from robotNavigation import RobotNavigation
import random

scale = int(sys.argv[1])
width = 10 * scale
height = 10 * scale

# discount factor
gamma = 0.9
# the time step that the agent receives the response
responseTime = 10 * scale

queries = []
for _ in xrange(6):
  x = int(width * random.random())
  y = int(height * random.random())
  queries.append((x, y))

def main():
  rewards0 = util.Counter()
  rewardList0 = [0, 0, 2, 2, -2, -2]
  for _ in xrange(6): rewards0[queries[_]] = rewardList0[_]

  rewards1 = util.Counter()
  rewardList1 = [0, 0, -2, -2, 2, 2]
  for _ in xrange(6): rewards1[queries[_]] = rewardList1[_]

  rewardSet = [rewardGen(rewards0), rewardGen(rewards1)]
  rewardNum = len(rewardSet)
  initialPhi = [1.0 / rewardNum] * rewardNum

  if sys.argv[2] == 'JQTP':
    Agent = JointQTPAgent
  elif sys.argv[2] == 'AQTP':
    Agent = IterativeQTPAgent
  else:
    raise Exception("Unknown Agent " + sys.argv[2])

  cmp = RobotNavigation(queries, rewardSet[0], gamma, responseTime, width, height)
  agent = Agent(cmp, rewardSet, initialPhi, relevance, gamma=gamma)
 
  ret, qValue, time = Experiment(cmp, agent, gamma, rewardSet, horizon=20 * scale)
  print ret
  print qValue
  print time

def relevance(fState, query):
  return abs(fState[0] - query[0]) + abs(fState[1] - query[1]) < 10 * scale

def rewardGen(rewards): 
  def rewardFunc(s):
    if s in rewards.keys():
      return rewards[s]
    else:
      return 0
  return rewardFunc

if __name__ == '__main__':
  main()
