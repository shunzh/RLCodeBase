from pymprog import *
import easyDomains
import pprint
from QTPAgent import QTPAgent

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

class MILPAgent(QTPAgent):
  def learn(self):
    args = easyDomains.convert(cmp)
    args['R'] = self.rewardSet
    args['psi'] = self.initialPhi
    args['maxV'] = [0]
    rewardCandNum = len(self.rewardSet)

    q = []
    # k is the number of possible responses of a query
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

    # now q is a set of policy queries


if __name__ == '__main__':
  rockDomain()