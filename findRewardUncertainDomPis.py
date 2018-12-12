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
  alpha = mdpR['alpha']

  # compute the optimal policy under mdpH
  value, humanPi = lp.lpDual(**mdpH)

  # compute the set of state, action pairs that have different transitions under mdpH and mdpH
  domPis = [humanPi]
  oldDomPis = []

  # repeat until domPis converges
  while len(domPis) > len(oldDomPis):
    for (s, a) in delta:
      # create a policy that takes action a in state s instead of pi(s)
      
      # first, reduce the occupancy on the whole mdp
      newPi = {sa: (1 - alpha(s)) * value for (sa, value) in humanPi.items()}
      # then add occupancy of alpha(s) on (s, a) back
      # and recursively add the occupancy of the following states back
      addOccupancy(mdpR, newPi, 1, s, a)
      
      # see whether we find a reward function so that newPi is better than humanPi


def addOccupancy(mdp, pi, occ, s, a=None):
  """
  add occupancy to s, a, recursively
  if a == None, then a := pi(s)
  """
  # recursion stop criterion
  if occ < 0.001: return

  if a == None:
    for ap in mdp['A']:
      if pi[s, ap] > 0:
        pi[s, ap] += occ * pi[s, ap] 

        [addOccupancy(mdp, pi, sp, mdp['gamma'] * occ) for sp in mdp['S'] if mdp['T'](s, a, sp) > 0]
  else:
    pi[(s, a)] += occ

    [addOccupancy(mdp, pi, sp, mdp['gamma'] * occ) for sp in mdp['S'] if mdp['T'](s, a, sp) > 0]

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
