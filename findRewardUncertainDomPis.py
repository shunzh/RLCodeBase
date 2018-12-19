import random
import numpy
import easyDomains
import copy
import lp
import config
from easyDomains import occupancyAdd

try:
  from pycpx import CPlexModel, CPlexException
except ImportError:
  print "can't import CPlexModel"

def findUndominatedReward(mdpH, mdpR, newPi, humanPi, localDifferentPis, domPis):
  """
  Implementation of the linear programming problem (Eq.2) in report 12.5
  Returns the objective value and a reward function (which is only useful when the obj value is > 0)
  
  newPi is \hat{\pi} in the linear programming problem in the report.
  The robot tries to see if there exists a reward function where newPi is better than the best policy in domPis.
  """
  m = CPlexModel()
  if not config.VERBOSE: m.setVerbosity(0)
  
  S = mdpH.S
  robotA = mdpR.A
  humanA = mdpH.A

  # index of states and actions
  Sr = range(len(S))
  robotAr = range(len(robotA))
  humanAr = range(len(humanA))
  
  r = m.new(len(S), lb=-1, ub=1, name='r')
  z = m.new(name='z') # when the optimal value is attained, z = \max_{domPi \in domPis} V^{domPi}_r

  for domPi in domPis:
    m.constrain(z >= sum([domPi[S[s], robotA[a]] * r[s] for s in Sr for a in robotAr]))
  
  # make sure r is consistent with humanPi
  for s in S:
    for a in humanA:
      humanAlterPi = localDifferentPis[s, a]
      m.constrain(sum((humanPi[S[sp], humanA[ap]] - humanAlterPi[S[sp], humanA[ap]]) * r[sp] for sp in Sr for ap in humanAr) >= 0)
    
  # maxi_r { V^{newPi}_r - \max_{domPi \in domPis} V^{domPi}_r }
  m.maximize(sum(newPi[S[s], robotA[a]] * r[s] for s in Sr for a in robotAr) - z)
  
  obj = sum([newPi[S[s], robotA[a]] * m[r][s] for s in Sr for a in robotAr]) - m[z]

  # the reward function has the same values for same states, but need to convert back to the S x A space
  rFunc = lambda s, a: m[r][Sr.index(s)]

  print 'obj', obj
  print 'newPi', newPi
  print 'z', m[z], 'r', m[r]
  raw_input()

  return obj, rFunc

def findDomPis(mdpH, mdpR, delta):
  """
  Implementation of algorithm 1 in report 12.5
  
  mdpH, mdpR: both agents' mdps. now we assume that they are only different in the action space: the robot's action set is a superset of the human's.
  delta: the actions that the robot can take and the human cannot.
  """
  # compute the set of state, action pairs that have different transitions under mdpH and mdpH
  S = mdpH.S
  robotA = mdpR.A
  T = mdpR.T # note that the human and the robot have the same transition probabilities. The robot just has more actions
  gamma = mdpH.gamma
 
  # find the occupancy of policy humanPi from any state
  occupancies = {}
  
  mdpLocal = copy.deepcopy(mdpH)
  for s in S:
    mdpLocal.resetInitialState(s)
    objValue, pi = lp.lpDual(mdpLocal)
    
    for (deltaS, deltaA) in delta:
      # the human is unbale to take this action, make sure here
      assert (deltaS, deltaA) not in pi.keys()
      pi[deltaS, deltaA] = 0
    occupancies[s] = pi
  # find the occupancy with uniform initial state distribution
  averageHumanOccupancy = {}
  for s0 in S:
    # passing mdpR because we need all actions
    occupancyAdd(mdpR, averageHumanOccupancy, occupancies[s0], 1.0 / len(S))

  # find the policies that are different from $\pi^*_\H$ only in one state
  localDifferentPis = {}
  for diffS in S:
    for diffA in robotA:
      pi = copy.deepcopy(averageHumanOccupancy)
      # remove the original occupancy
      occupancyAdd(mdpR, pi, occupancies[diffS], - 1.0 / len(S))
      # add action (diffS, diffA)
      occupancyAdd(mdpR, pi, {(diffS, diffA): 1}, 1.0 / len(S))

      # update the occupancy of states that can be reached by taking diffA in diffS
      for sp in S:
        if T(diffS, diffA, sp) > 0:
          occupancyAdd(mdpR, pi, occupancies[sp], 1.0 / len(S) * gamma * T(diffS, diffA, sp))
          
      localDifferentPis[diffS, diffA] = pi

  print 'average human', averageHumanOccupancy
  domPis = [averageHumanOccupancy]
  domPiAdded = True
 
  domRewards = [] # the optimal policies under which are dominating policies
  # repeat until domPis converges

  while domPiAdded:
    domPiAdded = False
    for (s, a) in delta:
      # change the action in state s from \pi^*_\H(s) to a
      newPi = localDifferentPis[s, a]

      print s, a
      objValue, r = findUndominatedReward(mdpH, mdpR, newPi, averageHumanOccupancy, localDifferentPis, domPis)
      
      if objValue > 0.0001: # resolve numerical issues
        domRewards.append(r)
      
        # find the corresponding optimal policy and add to the set of dominating policies
        mdpR.r = r
        _, newDompi = lp.lpDual(mdpR)

        domPis.append(newDompi)
        
        domPiAdded = True
    
    print domPis
  
  return domPis

def printReward(S, A, r):
  for s in S:
    print s, [r(s, a) for a in A]
  
def adjustOccupancy(mdp, pi, occ, s, a=None):
  """
  DUMMY?

  add occ to s, a, recursively, and stop when reaching the terminal state or the occupancy is too small
  if a == None, then a := pi(s)
  """
  # recursion stop criterion
  if mdp['terminal'](s) or abs(occ) < 0.001: return

  if a == None:
    for ap in mdp.A:
      if pi[s, ap] > 0:
        pi[s, ap] += occ * pi[s, ap] 

        [adjustOccupancy(mdp, pi, sp, mdp.gamma * occ) for sp in mdp.S if mdp.T(s, a, sp) > 0]
  else:
    pi[(s, a)] += occ

    [adjustOccupancy(mdp, pi, sp, mdp.gamma * occ) for sp in mdp.S if mdp.T(s, a, sp) > 0]

def toyMDP():
  """
  Starting from state 0, the robot has two actions to reach state 1 and 2, respectively, and then stay there.
  """
  mdp = easyDomains.SimpleMDP()

  mdp.S = range(3)
  mdp.A = range(2)

  tDict = {(0, 0): 1, (0, 1): 2, (1, 0): 1, (1, 1): 1, (2, 0): 2, (2, 1): 2}
  mdp.T = lambda s, a, sp: tDict[s, a] == sp # deterministic transitions

  mdp.r = lambda s, a: s == 1 # only state 1 has positive reward

  mdp.gamma = .5

  mdp.terminal = lambda _: False

  mdp.alpha = lambda s: 1.0 / len(mdp.S)
  
  return mdp

def experiment():
  # the human's mdp
  #rewardSparsity = .2
  #mdpR = easyDomains.randMDP(states, robotActions, rewardSparsity)
  
  mdpR = toyMDP()

  mdpH = copy.deepcopy(mdpR)

  # restrict the human's actions space
  # the human can only reach state 1. state 2 is unreachable from state 0.
  mdpH.A = range(1)
  
  delta = [(s, a) for s in mdpR.S for a in mdpR.A if a not in mdpH.A]

  findDomPis(mdpH, mdpR, delta)
  

if __name__ == '__main__':
  rnd = 0
  
  random.seed(rnd)
  # not necessarily using the following packages, but just to be sure
  numpy.random.seed(rnd)
  
  experiment()
