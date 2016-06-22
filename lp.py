from pymprog import *
import easyDomains

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

  # decision variables
  x = var(xrange(len(S) * len(A)), 'x', bounds=(0, None))
  z = var(xrange(rLen), 'z', bool)
  
  y = var(xrange(rLen))

  # obj
  minimize(sum([psi[i] * y[i] for i in xrange(rLen)]))
  
  # constraints on y
  st([y[i] <= sum([x(s, a) * R[i](s, a) for s in S for a in A]) - maxV[i] + z[i] * M for i in xrange(rLen)])
  st([y[i] <= (1 - z[i]) * M for i in xrange(rLen)])
  
  # constraints on x (valid occupancy)
  for sp in S:
    if sp == s0:
      st(sum([x(sp, ap) for ap in A]) == 1)
    else:
      st(sum([x(sp, ap) for ap in A]) == sum([x(s, a) * T(s, a, sp) for s in S for a in A]))

if __name__ == '__main__':
  # pass domain here
  args = easyDomains.getRockDomain(10, 5, 10)
  
  for i in range(k):
    if i == 0: args['maxV'] = 0
    else: args['maxV'] = 

    lp(**args)
