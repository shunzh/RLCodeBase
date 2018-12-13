import random
import numpy
import easyDomains
import copy
import lp
import config

try:
  from pycpx import CPlexModel, CPlexException
except ImportError:
  print "can't import CPlexModel"

def findUndominatedReward(mdp, newPi, humanPi, domPis):
  """
  Implementation of the linear programming problem (Eq.2) in report 12.5
  Returns the objective value and a reward function (which is only useful when the obj value is > 0)
  
  newPi is \hat{\pi} in the linear programming problem in the report.
  The robot tries to see if there exists a reward function where newPi is better than the best policy in domPis.
  """
  m = CPlexModel()
  if not config.VERBOSE: m.setVerbosity(0)
  
  S = mdp['S']
  A = mdp['A']
  T = mdp['T']
  gamma = mdp['gamma']
  alpha = mdp['alpha']

  # useful constants
  Sr = range(len(S))
  Ar = range(len(A))
  
  r = m.new(len(S), lb=-1, ub=1, name='r')
  z = m.new(name='z') # when the optimal value is attained, z = \max_{domPi \in domPis} V^{domPi}_r

  for domPi in domPis:
    m.constrain(z >= sum([domPi[S[s], A[a]] * r[s] for s in Sr for a in Ar]))
  
  # make sure r is consistent with humanPi
  for s in Sr:
    localDiffPi = 
    m.constrain(sum([(humanPi[S[s], A[a]] - localDiffPi[S[s], A[a]]) * r[s] for s in Sr for a in Ar]) >= 0)
    
  try:
    # maxi_r { V^{newPi}_r - \max_{domPi \in domPis} V^{domPi}_r }
    obj = m.maximize(sum([newPi[S[s], A[a]] * r[s] for s in Sr for a in Ar]))
  except CPlexException as err:
    print 'Exception', err
    return None, {}

  # the reward function has the same values for same states, but need to convert back to the S x A space
  rFunc = lambda s, a: m[r][Sr.index(s)]
  return obj, rFunc

def findDomPis(mdpH, mdpR, humanPi, delta):
  """
  Implementation of algorithm 1 in report 12.5
  
  mdpH, mdpR: both agents' mdps. now we assume that they are only different in the action space: the robot's action set is a superset of the human's.
  delta: the actions that the robot can take and the human cannot.
  
  FIXME r is actually defined in both MDP. just don't use them since the robot has no such knowledge
  """
  alpha = mdpH['alpha'] # same for both mdps anyway

  # compute the set of state, action pairs that have different transitions under mdpH and mdpH
  domPis = [humanPi]
  oldDomPis = []

  # repeat until domPis converges
  while len(domPis) > len(oldDomPis):
    for (s, a) in delta:
      newPi = copy.deepcopy(humanPi)

      # then add occupancy of alpha(s) on (s, a) back
      # and recursively add the occupancy of the following states back
      adjustOccupancy(mdpR, newPi, -alpha(s), s)
      adjustOccupancy(mdpR, newPi, alpha(s), s, a)
 

def adjustOccupancy(mdp, pi, occ, s, a=None):
  """
  add occ to s, a, recursively, and stop when reaching the terminal state or the occupancy is too small
  if a == None, then a := pi(s)
  """
  # recursion stop criterion
  if mdp['terminal'](s) or abs(occ) < 0.001: return

  if a == None:
    for ap in mdp['A']:
      if pi[s, ap] > 0:
        pi[s, ap] += occ * pi[s, ap] 

        [adjustOccupancy(mdp, pi, sp, mdp['gamma'] * occ) for sp in mdp['S'] if mdp['T'](s, a, sp) > 0]
  else:
    pi[(s, a)] += occ

    [adjustOccupancy(mdp, pi, sp, mdp['gamma'] * occ) for sp in mdp['S'] if mdp['T'](s, a, sp) > 0]

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
  
  value, humanPi = lp.lpDual(**mdpH) 

  findDomPis(mdpH, mdpR, humanPi, delta)
  

if __name__ == '__main__':
  rnd = 0
  
  random.seed(rnd)
  # not necessarily using the following packages, but just to be sure
  numpy.random.seed(rnd)
