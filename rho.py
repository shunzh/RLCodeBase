# needs refactoring

import util
from pprint import pprint
import numpy

rhoMin = util.Counter()
rhoMax = util.Counter()

size = 5
rho = 0.8
noise = 0.5

actions = [(-1, 1), (0, 1), (1, 1)]
rhoMax[(0, size - 1)] = 1
rhoMax[(size - 1, size - 1)] = 1
for i in range(size): rhoMin[(i, 0)] = numpy.inf
rhoMin[(size / 2, 0)] = rho

def getActions(s):
  if s[0] == 0: return actions[1:]
  elif s[0] == size - 1: return actions[:-1]
  else: return actions

def getNextState(s, a):
  nextS = [s[0] + a[0], s[1] + a[1]]
  if nextS[0] < 0: nextS[0] = 0
  elif nextS[0] > size - 1: nextS[0] = size - 1
  return tuple(nextS)

def transit(s, a):
  left = getNextState(s, (-1, 1))
  mid = getNextState(s, (0, 1))
  right = getNextState(s, (1, 1))

  if a == (-1, 1): return {left: 1}
  elif a == (0, 1):
    if left == mid:
      return {left: 1}
    else:
      return {left: noise, mid: 1 - noise}
  elif a == (1, 1): 
    if mid == right:
      return {mid: 1}
    else:
      return {mid: noise, right: 1 - noise}

# rhomax
for j in reversed(range(size - 1)):
  for i in range(size):
    values = []
    for a in getActions((i, j)):
      transitions = transit((i, j), a)
      value = sum([p * rhoMax[s] for (s, p) in transitions.items()])
      values.append(value)
    rhoMax[(i, j)] = max(values)

rhoMax[(0, 0)] = 0
rhoMax[(1, 0)] = 0
rhoMax[(3, 0)] = 0
rhoMax[(4, 0)] = 0
rhoMax[(0, 1)] = 0
rhoMax[(4, 1)] = 0

# rhomin
for j in range(1, size):
  for i in range(size):
    values = []
    for k in range(size):
      for a in getActions((k, j - 1)):
        transitions = transit((k, j - 1), a)
        if (i, j) in transitions.keys():
          values.append((rhoMin[(k, j - 1)] - sum([p * rhoMax[s] for (s, p) in transitions.items()])\
                         + transitions[(i, j)] * rhoMax[(i, j)])\
                        / transitions[(i, j)])
    rhoMin[(i, j)] = max(min(values), 0)

for j in reversed(range(size)):
  for i in range(size):
    print rhoMax[(i, j)],
  print
for j in reversed(range(size)):
  for i in range(size):
    print rhoMin[(i, j)],
  print

for j in reversed(range(size)):
  for i in range(size):
    if rhoMax[(i, j)] >= rhoMin[(i, j)]: print 1,
    else: print 0,
  print
