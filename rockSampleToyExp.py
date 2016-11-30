from tabularNavigationExp import experiment

if __name__ == '__main__':
  # use rockNum == 0 to represent a test case
  rewardCandNum = 3

  """
  def r1(s, a):
    if s == (0, 0) and a == (1, 0): return 0.9
    elif s == (0, 1) and a == (0, 1): return 0.6
    else: return 0
  def r2(s, a):
    if s == (0, 1) and a == (1, 0): return 1
    elif s == (0, 1) and a == (0, 1): return 0.6
    else: return 0
  def r3(s, a):
    if s == (0, 0) and a == (1, 0): return 0.45
    elif s == (0, 1) and a == (1, 0): return 0.5
    elif s == (0, 1) and a == (0, 1): return 0.6
    else: return 0
  rewardSet = [r1, r2, r3]
  """

  r1 = lambda s, a: s == (2, 10)
  r2 = lambda s, a: s == (1, 10)
  r3 = lambda s, a: s == (3, 10)
  rewardSet = [r1, r2, r3]
  
  initialPhi = [1.0 / rewardCandNum] * rewardCandNum

  experiment(cmp, rewardSet, initialPhi)