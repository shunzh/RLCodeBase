import util
from tabularNavigation import PuddleWorld
from tabularNavigationExp import experiment
import random
import getopt
import sys
import tabularNavigationExp
import numpy as np

if __name__ == '__main__':
  width = 21
  height = 21
  # the time step that the agent receives the response
  responseTime = 0
  horizon = height - 1
  
  Domain = PuddleWorld

  try:
    opts, args = getopt.getopt(sys.argv[1:], tabularNavigationExp.flags)
  except getopt.GetoptError:
    raise Exception('Unknown flag')
  for opt, arg in opts:
    if opt == '-t':
      rockType = arg
    elif opt == '-r':
      random.seed(int(arg))
 
  # rocks have different values
  rocks = []

  #puddleNum = 3; puddleLocs = [(width / 4, height / 4), (width / 2, height / 4), (width / 2, height / 2)]
  puddleNum = 6
  puddleSize = 2
  rockPerPuddle = 2
  puddles = []
  for _ in xrange(puddleNum):
    loc = (random.randint(0, width - puddleSize - 1), random.randint(0, height - puddleSize - 1))
    #loc = puddleLocs[_]
    puddle = [(x, y) for x in range(loc[0], loc[0] + puddleSize) for y in range(loc[1], loc[1] + puddleSize)]
    puddles.append(puddle)
    for rockID in xrange(rockPerPuddle):
      rocks.append(random.choice(puddle))

  rewardBasic = util.Counter()
  for id in xrange(len(rocks)):
    rewardBasic[rocks[id]] = id + 1

  # TODO use reward gen
  # reward cand indicates belief on where the puddle is
  rewardCandNum = puddleNum
  rewards = []
  for candId in xrange(rewardCandNum):
    reward = rewardBasic.copy()
    puddle = puddles[candId]
    for loc in puddle: reward[loc] = -10
    rewards.append(reward)

  initialPhi = [1.0 / len(rewards)] * len(rewards)
  
  terminalReward = util.Counter()
  for x in range(width / 3) + range(width * 2 / 3, width): terminalReward[(x, height - 1)] = -1000

  allPuddles = sum(puddles, []) # concatenate its elements (which are lists)
  
  experiment(Domain, width, height, responseTime, horizon, rewardCandNum, allPuddles, rewards, initialPhi, terminalReward)