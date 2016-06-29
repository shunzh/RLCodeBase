from pycpx import CPlexModel
import easyDomains
import pprint
from QTPAgent import ActiveSamplingAgent
from cmp import QueryType
import scipy.stats
import random
import config

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
  m = CPlexModel()
  if not config.VERBOSE: m.setVerbosity(0)

  # useful constants
  rLen = len(R)
  M = 10000 # a large number
  Sr = range(len(S))
  Ar = range(len(A))
  
  # decision variables
  x = m.new((len(S), len(A)), lb=0, ub=1, name='x')
  z = m.new(rLen, vtype=bool, name='z')
  y = m.new(rLen, name='y')

  # constraints on y
  m.constrain([y[i] <= sum([x[s, a] * R[i](S[s], A[a]) for s in Sr for a in Ar]) - maxV[i] + z[i] * M for i in xrange(rLen)])
  m.constrain([y[i] <= (1 - z[i]) * M for i in xrange(rLen)])
  
  # constraints on x (valid occupancy)
  for sp in Sr:
    if S[sp] == s0:
      m.constrain(sum([x[sp, ap] for ap in Ar]) == 1)
    else:
      m.constrain(sum([x[sp, ap] for ap in Ar]) == sum([x[s, a] * T(S[s], A[a], S[sp]) for s in Sr for a in Ar]))
  
  # obj
  obj = m.maximize(sum([psi[i] * y[i] for i in xrange(rLen)]))

  if config.VERBOSE:
    print 'obj', obj
    print 'x', m[x]
    print 'y', m[y]
    print 'z', m[z]
  
  # build occupancy as S x A -> x[.,.]
  return {(S[s], A[a]): m[x][s, a] for s in Sr for a in Ar}

def computeValue(pi, r, S, A):
  sum = 0
  for s in S:
    for a in A:
      sum += pi[s, a] * r(s, a)
  return sum

def rockDomain():
  size = 10
  numRocks = 3
  rewardCandNum = 3
  args = easyDomains.getRockDomain(size, numRocks, rewardCandNum, fixedRocks=True)
  k = 3 # number of responses
  
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

    hList = []
    for s in args['S']:
      hValue = 0
      for a in args['A']:
        bins = [0] * 10
        for pi in q:
          id = min([int(10 * pi[s, a]), 9])
          bins[id] += 1
        hValue += scipy.stats.entropy(bins)
      hList.append((s, hValue))

    hList = sorted(hList, reverse=True, key=lambda _: _[1])

def toyDomain():
  args = easyDomains.getChainDomain(10)
  args['maxV'] = [0]
  lp(**args)

class MILPAgent(ActiveSamplingAgent):
  def learn(self):
    args = easyDomains.convert(self.cmp, self.rewardSet, self.phi)
    args['maxV'] = [0]
    rewardCandNum = len(self.rewardSet)

    # now q is a set of policy queries
    q = []
    for i in range(len(args['A'])):
      if i == 0:
        args['maxV'] = [0] * rewardCandNum
      else:
        # find the optimal policy so far that achieves the best on each reward candidate
        args['maxV'] = []
        for rewardId in xrange(rewardCandNum):
          args['maxV'].append(max([computeValue(pi, args['R'][rewardId], args['S'], args['A']) for pi in q]))

      x = lp(**args)
      """
      for s in args['S']:
        for a in args['A']:
          if x[s, a].primal > 0: print s, a, x[s, a]
      """
      q.append(x)

    if self.queryType == QueryType.ACTION:
      hList = []
      for s in args['S']:
        hValue = 0
        for a in args['A']:
          # for all possible responses of the action query
          bins = [0] * 10
          for pi in q:
            id = min([int(10 * pi[s, a]), 9])
            bins[id] += 1
          hValue += scipy.stats.entropy(bins)
          #print s, a, bins
        hList.append((s, hValue))

      hList = sorted(hList, reverse=True, key=lambda _: _[1])
      #print hList
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
  config.VERBOSE = True

  #rockDomain()
  toyDomain()
