import random
rewardNum = 10
numConfigs = 3

for _ in xrange(300):
  rewardCandidates = {(i, j): random.random() * 3 for i in range(rewardNum)\
                                                  for j in range(numConfigs)}
  meanRewards = [sum([rewardCandidates[(i, j)] for i in range(rewardNum)])\
                                               for j in range(numConfigs)]
  maxMeanConfig = max(range(numConfigs), key=lambda j: meanRewards[j])
  # assume first reward is the true reward

  v = rewardCandidates[(0, maxMeanConfig)]

  rFile = open('results', 'a')
  rFile.write(str(v) + '\n')
  rFile.close()