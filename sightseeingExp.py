from QTPAgent import IterativeQTPAgent, JointQTPAgent
from CMPExp import Experiment
import util
from sightseeing import Sightseeing
import random
import sys

# arguments: sightseeingExp.py scale algorithm

scale = int(sys.argv[1])
# it is like a corridor from the driver's view
width = 8 * 4 * scale
height = 3

# discount factor
gamma = 0.9
# the time step that the agent receives the response
responseTime = 16 * scale

queries = []
for _ in xrange(10 * scale):
  x = int(width * random.random())
  y = int(height * random.random())
  queries.append((x, y, 1, 0))
  queries.append((x, y, -1, 0))

def main():
  rewards = []
  rewardNum = 5

  for _ in xrange(rewardNum):
    # for each reward candidate, 5 possible sights
    reward = util.Counter()
    for idx in xrange(5 * scale):
      query = random.choice(queries)
      x, y, dir, status = query
      reward[(x, y)] = 1
    rewards.append(reward)

  rewardSet = [rewardGen(reward) for reward in rewards]
  initialPhi = [1.0 / rewardNum] * rewardNum

  if sys.argv[2] == 'JQTP':
    Agent = JointQTPAgent
  elif sys.argv[2] == 'AQTP':
    Agent = IterativeQTPAgent
  else:
    raise Exception("Unknown Agent " + sys.argv[2])

  cmp = Sightseeing(queries, rewardSet[0], gamma, responseTime, width, height)
  agent = Agent(cmp, rewardSet, initialPhi, gamma=gamma)
 
  ret, qValue, timeElapsed = Experiment(cmp, agent, gamma, rewardSet)
  print ret
  print qValue
  print timeElapsed

def rewardGen(rewards): 
  def rewardFunc(s):
    x, y, dir, status = s
    if status == 1:
      if (x, y) in rewards.keys():
        return rewards[(x, y)]
      else:
        return -1
    elif s[0] == 0 and s[1] == 0 and s[2] != 0:
      return 2
    else:
      return 0
  return rewardFunc

if __name__ == '__main__':
  main()
