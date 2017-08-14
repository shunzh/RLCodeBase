try:
  from pycpx import CPlexModel
except ImportError:
  print "can't import CPlexModel"
import easyDomains
import scipy.stats
import config
import util

def lp(S, A, r, T, s0):
  """
  Solve the LP problem to find out the optimal occupancy
  
  Args:
    S: state set
    A: action set
    r: reward
    T: transition function
    s0: init state
  """
  m = CPlexModel()
  if not config.VERBOSE: m.setVerbosity(0)

  # useful constants
  Sr = range(len(S))
 
  v = m.new(len(S), name='v')

  for s in Sr:
    for a in A:
      m.constrain(v[s] >= r(S[s], a) + sum(v[sp] * T(S[s], a, S[sp]) for sp in Sr))
  
  # obj
  obj = m.minimize(v[s0])
  ret = util.Counter()
  for s in Sr:
    ret[S[s]] = m[v][s]
  return ret

def lpDual(S, A, r, T, s0, gamma=1, constraints={}, positiveConstraints={}):
  """
  Solve the dual problem of lp
  Same arguments
  
  Note that this is a lower level function that does not consider feature extraction.
  r should be a reward function, not a reward parameter.
  """
  m = CPlexModel()
  if not config.VERBOSE: m.setVerbosity(0)

  # useful constants
  Sr = range(len(S))
  Ar = range(len(A))
 
  x = m.new((len(S), len(A)), lb=0, ub=1, name='x')

  # make sure x is a valid occupancy
  for sp in Sr:
    # x (x(s) - \gamma * T) = \sigma
    m.constrain(sum([x[s, a] * ((s == sp) - gamma * T(S[s], A[a], S[sp])) for s in Sr for a in Ar]) == (S[sp] == s0))
    #print S[sp], [(S[s], A[a]) for s in Sr for a in Ar if T(S[s], A[a], S[sp]) > 0]
  
  # == constraints
  for (s, a), occ in constraints.items():
    m.constrain(x[S.index(s), A.index(a)] == occ)

  # >= constraints
  for (s, a), occ in positiveConstraints.items():
    m.constrain(x[S.index(s), A.index(a)] >= occ)

  # obj
  obj = m.maximize(sum([x[s, a] * r(S[s], A[a]) for s in Sr for a in Ar]))
  #for sp in Sr:
  #  if S[sp] != s0:
  #    print sp, sum([m[x][sp, ap] for ap in Ar]) - gamma * sum([m[x][s, a] * T(S[s], A[a], S[sp]) for s in Sr for a in Ar])

  return obj, {(S[s], A[a]): m[x][s, a] for s in Sr for a in Ar}

def milp(S, A, R, T, s0, psi, maxV):
  """
  Solve the MILP problem in greedy construction of policy query
  
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
  m.constrain([y[i] <= sum([x[s, a] * R[i](S[s], A[a]) for s in Sr for a in Ar]) - maxV[i] + (1 - z[i]) * M for i in xrange(rLen)])
  m.constrain([y[i] <= z[i] * M for i in xrange(rLen)])
  
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
  # z[i] == 1 then this policy is better than maxV on the i-th reward candidate
  res = util.Counter()
  for s in Sr:
    for a in Ar:
      res[S[s], A[a]] = m[x][s, a] 
  return res

def computeObj(q, psi, S, A, R):
  rLen = len(R)
  obj = 0

  for i in xrange(rLen):
    values = [computeValue(pi, R[i], S, A) for pi in q]
    obj += psi[i] * max(values)
    #print filter(lambda _: values[_] == max(values), range(len(q))), max(values)
  
  return obj

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

    x = milp(**args)
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
  milp(**args)


if __name__ == '__main__':
  config.VERBOSE = True

  #rockDomain()
  toyDomain()
