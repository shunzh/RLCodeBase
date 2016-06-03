import util
from tabularNavigation import PuddleWorld
from tabularNavigationExp import experiment
import random

if __name__ == '__main__':
  width = 21
  height = 21
  # the time step that the agent receives the response
  responseTime = 0
  horizon = height
  
  Domain = PuddleWorld
  
  # rocks have different values
  rocksNum = 20
  rocks = []
  for _ in xrange(rocksNum):
    loc = (random.randint(0, width - 1), random.randint(0, height - 1))
    if not loc in rocks:
      rocks.append(loc)
  rewardBasic = util.Counter()
  for id in xrange(len(rocks)):
    rewardBasic[rocks[id]] = id + 1
    
  puddleNum = 5
  puddleSize = 3
  puddles = []
  for _ in xrange(puddleNum):
    loc = (random.randint(0, width - puddleSize - 1), random.randint(0, height - puddleSize - 1))
    if not loc in puddles:
      puddles.append([(x, y) for x in range(loc[0], loc[0] + puddleSize) for y in range(loc[1], loc[1] + puddleSize)])

  # reward cand indicates belief on where the puddle is
  rewardCandNum = puddleNum
  rewards = []
  for candId in xrange(rewardCandNum):
    reward = rewardBasic.copy()
    puddle = puddles[candId]
    for loc in puddle: reward[loc] -= 10
    rewards.append(reward)

  initialPhi = [1.0 / len(rewards)] * len(rewards)
  
  terminalReward = util.Counter()

  allPuddles = sum(puddles, []) # concatenate its elements (which are lists)
  
  experiment(Domain, width, height, responseTime, horizon, rewardCandNum, allPuddles, rewards, initialPhi, terminalReward)