from pymprog import *
import easyDomains
import pprint
from QTPAgent import ActiveSamplingAgent
from cmp import QueryType
import scipy.stats
import random

def lp(S, A, R, T, s0, psi, maxV):
  """
  Args:
    S: state set
    A: action set
    R: reward candidate set
    T: transition function
    s0: init state
    psi: prior belief on rewards
    maxV: maxV[i] = max_{\pi \in q} V_{r_i}^\pi
  """
  # useful constants
  rLen = len(R)
  M = 10000 # a large number
  SA = iprod(S, A)
  
  beginModel()
  # decision variables
  x = var(SA, 'x', bounds=(0, 1))
  z = var(xrange(rLen), 'z', bool)
  y = var(xrange(rLen), 'y')

  # obj
  maximize(sum([psi[i] * y[i] for i in xrange(rLen)]))
  
  # constraints on y
  st([y[i] <= sum([x[s, a] * R[i](s, a) for s in S for a in A]) - maxV[i] + z[i] * M for i in xrange(rLen)])
  st([y[i] <= (1 - z[i]) * M for i in xrange(rLen)])
  
  # constraints on x (valid occupancy)
  for sp in S:
    if sp == s0:
      st(sum([x[sp, ap] for ap in A]) == 1)
    else:
      st(sum([x[sp, ap] for ap in A]) == sum([x[s, a] * T(s, a, sp) for s in S for a in A]))
  
  solvopt(integer='advanced')
  solve()
  print 'Obj =', vobj()
  print y, z
  
  #pprint.pprint(x)
  endModel()
  return x

def computeValue(pi, r, S, A):
  sum = 0
  for s in S:
    for a in A:
      sum += pi[s, a].primal * r(s, a)
  return sum

def rockDomain():
  size = 10
  numRocks = 5
  rewardCandNum = 10
  args = easyDomains.getRockDomain(size, numRocks, rewardCandNum)
  k = 5 # number of responses
  
  q = [] # query set

  for i in range(k):
    if i == 0:
      args['maxV'] = [0] * rewardCandNum
    else:
      # find the optimal policy so far that achieves the best on each reward candidate
      args['maxV'] = []
      for rewardId in xrange(rewardCandNum):
        args['maxV'].append(max([computeValue(pi, args['R'][rewardId], args['S'], args['A']) for pi in q]))

    x = lp(**args)
    q.append(x)

def toyDomain():
  args = easyDomains.getChainDomain(10)
  args['maxV'] = [0]
  lp(**args)

class MILPAgent(ActiveSamplingAgent):
  def learn(self):
    args = easyDomains.convert(cmp, self.rewardSet, self.initialPhi)
    args['maxV'] = [0]
    rewardCandNum = len(self.rewardSet)

    # now q is a set of policy queries
    q = []
    for i in range(self.m):
      if i == 0:
        args['maxV'] = [0] * rewardCandNum
      else:
        # find the optimal policy so far that achieves the best on each reward candidate
        args['maxV'] = []
        for rewardId in xrange(rewardCandNum):
          args['maxV'].append(max([computeValue(pi, args['R'][rewardId], args['S'], args['A']) for pi in q]))

      x = lp(**args)
      q.append(x)

    if self.queryType == QueryType.ACTION:
      hList = []
      for s in self.cmp.getStates():
        hValue = 0
        for a in self.cmp.getPossibleActions(self.cmp.state):
          bins = [0] * 10
          for pi in q:
            id = min(10 * pi(s, a), 9)
            bins[id] += 1
          hValue += scipy.stats.entropy(bins)
        hList.append((s, hValue))

      hList = sorted(hList, reverse=True, key=lambda _: _[1])
      hList = hList[:self.m]
    else:
      raise Exception('Query type not implemented for MILP.')

    qList = []
    for q, h in hList:
      pi, qValue = self.optimizePolicy(q)
      qList.append((q, pi, qValue))

    maxQValue = max(map(lambda _:_[2], qList))
    qList = filter(lambda _: _[2] == maxQValue, qList)

    return random.choice(qList)

if __name__ == '__main__':
  rockDomain()