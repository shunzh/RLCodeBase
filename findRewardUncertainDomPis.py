import random
import numpy
import easyDomains
import copy
import lp


def findUndominatedReward():
  """
  Implementation of the linear programming problem (Eq.2) in report 12.5
  """

def findDomPis(mdpH, mdpR, delta):
  """
  Implementation of algorithm 1 in report 12.5
  
  mdpH, mdpR: both agents' mdps. now we assume that they are only different in the action space: the robot's action set is a superset of the human's.
  delta: the actions that the robot can take and the human cannot.
  """
  # compute the optimal policy under mdpH
  value, humanPi = lp.lpDual(**mdpH)

  # compute the set of state, action pairs that have different transitions under mdpH and mdpH
  domPis = [humanPi]
  oldDomPis = []

  # repeat until domPis converges
  while len(domPis) > len(oldDomPis):
    for (s, a) in delta:
      newPi

def experiment():
  states = range(10)

  robotActions = range(4)
  humanActions = range(3)
  
  delta = [(s, a) for s in states for a in robotActions if a not in humanActions]

  rewardSparsity = .2

  # the human's mdp
  mdpR = easyDomains.randMDP(states, robotActions, rewardSparsity)

  mdpH = copy.deepcopy(mdpR)
  # restrict the human's actions space
  mdpH['A'] = humanActions
  
  findDomPis(mdpH, mdpR, delta)
  

if __name__ == '__main__':
  rnd = 0
  
  random.seed(rnd)
  # not necessarily using the following packages, but just to be sure
  numpy.random.seed(rnd)
