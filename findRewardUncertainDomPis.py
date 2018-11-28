import random
import numpy
import easyDomains


def findDomPis():
  numStates = 10
  numActions = 4
  rewardSparsity = .2
  
  numAdditionalActions = 1

  # the human's mdp
  mdpH = easyDomains.randMDP(numStates, numActions, rewardSparsity)
  # the robot has more capacity
  mdpR = easyDomains.addActionsToMDP(mdpH, numAdditionalActions)
  
  # find the optimal policy under mdpH
  
  
  # solving lp problems to see if we find better policies
  

if __name__ == '__main__':
  rnd = 0
  
  random.seed(rnd)
  # not necessarily using the following packages, but just to be sure
  numpy.random.seed(rnd)
