from QTPAgent import AlternatingQTPAgent, JointQTPAgent, RandomQueryAgent
from CMPExp import Experiment
import util
from sightseeing import Sightseeing
import random
import sys
import config
from cmp import QueryType
import getopt

def main():
  # discount factor
  gamma = 0.9
  # the time step that the agent receives the response
  responseTime = 10
  scale = 1

  try:
    opts, args = getopt.getopt(sys.argv[1:], "r:l:s:d:a:c:v")
  except getopt.GetoptError:
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-r':
      random.seed(int(arg))
    elif opt == '-l':
      responseTime = int(arg)
    elif opt == '-s':
      scale = int(arg)
    elif opt == '-d':
      gamma = float(arg)
    elif opt == '-a':
      agentName = arg
    elif opt == '-v':
      config.VERBOSE = True
 
  width = 40
  height = 3
  locations = [(random.randint(1, width - 1), random.randint(0, height - 1)) for _ in xrange(6 * scale)]
  # sort by x coordinate for convenience
  locations.sort(key=lambda _: _[0])
  queries = []
  for _ in xrange(6 * scale):
    x, y = locations[_]
    queries.append((x, y, 1, 0))
    queries.append((x, y, -1, 0))

  rewards = []
  rewardNum = 3

  # divide features by regions
  for _ in xrange(rewardNum):
    # for each reward candidate, 5 possible sights
    reward = util.Counter()
    for idx in xrange(2 * scale * _, 2 * scale * (_ + 1)):
      reward[locations[idx]] = 1
    rewards.append(reward)

  def relevance(fState, query):
    # see whether feature, query are relevant
    if fState[2] == 1:
      # forward
      if query[0] >= fState[0] and query[2] == 1:
        return True
    else:
      # backward
      if query[0] <= fState[0] and query[2] == -1:
        return True

  def rewardGen(rewards): 
    def rewardFunc(s):
      x, y, dir, status = s
      if status == 1:
        if (x, y) in rewards.keys():
          return rewards[(x, y)]
        else:
          return -1
      elif s[0] == 0 and s[1] == 0 and s[2] != 0:
        return 2
      else:
        return 0
    return rewardFunc

  rewardSet = [rewardGen(reward) for reward in rewards]
  initialPhi = [1.0 / rewardNum] * rewardNum
  if agentName == 'NQ':
    queries = [0] # make a dummy query set
    queryType = QueryType.NONE
  else:
    queryType = QueryType.POLICY

  cmp = Sightseeing(queries, random.choice(rewardSet), gamma, responseTime, width, height)

  if agentName == 'JQTP' or agentName == 'NQ':
    agent = JointQTPAgent(cmp, rewardSet, initialPhi, queryType, gamma)
  elif agentName == 'AQTP':
    agent = AlternatingQTPAgent(cmp, rewardSet, initialPhi, queryType, relevance, gamma)
  elif agentName == 'AQTP-NF':
    # don't filter query. Assume all queries are relevant.
    agent = AlternatingQTPAgent(cmp, rewardSet, initialPhi, queryType, lambda fS, q: True, gamma)
  elif agentName == 'AQTP-RS':
    agent = AlternatingQTPAgent(cmp, rewardSet, initialPhi, queryType, relevance, gamma, restarts=1)
  elif agentName == 'RQ':
    agent = RandomQueryAgent(cmp, rewardSet, initialPhi, queryType, gamma)
  elif agentName == 'TPNQ':
    agent = JointQTPAgent(cmp, rewardSet, initialPhi, queryType, gamma, queryIgnored=True)
  else:
    raise Exception("Unknown Agent " + agentName)

  ret, qValue, timeElapsed = Experiment(cmp, agent, gamma, rewardSet, queryType)
  print ret
  print qValue
  print timeElapsed

if __name__ == '__main__':
  main()
