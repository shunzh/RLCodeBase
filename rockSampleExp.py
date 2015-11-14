from QTPAgent import AlternatingQTPAgent, JointQTPAgent, RandomQueryAgent
from CMPExp import Experiment
import util
import sys
from rockSample import RockSample
import random
import getopt
import config
from cmp import QueryType

"""
Algorithms:
- E-JQTP
- AQTP
- AQTP-NF
- AQTP-RS (?)
- opt query
- random query
- no query

Setting:
- response time (axis)

Show:
- expectation Q
- computation time
- paired difference between JQTP and AQTP

TODO:
make this class more general, not just for rocksample exp
"""
def main():
  width = 20
  height = 20
  # the time step that the agent receives the response
  responseTime = 10
  horizon = 40
  objNum = 8

  # discount factor
  gamma = 0.9
  rewardCandNum = 3
  objNumPerFeat = 3
  obstacleEnabled = False

  agentName = 'JQTP'
  
  try:
    opts, args = getopt.getopt(sys.argv[1:], "r:l:s:d:a:ov")
  except getopt.GetoptError:
    print 'unknown flag encountered'
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-r':
      random.seed(int(arg))
    elif opt == '-l':
      responseTime = int(arg)
    elif opt == '-s':
      width = height = int(arg)
      horizon = width + height
    elif opt == '-d':
      gamma = float(arg)
    elif opt == '-a':
      agentName = arg
    elif opt == '-o':
      obstacleEnabled = True
    elif opt == '-v':
      config.VERBOSE = True
    
  queries = []
  rewards = []
  for _ in xrange(objNum):
    posCoin = random.random()
    if posCoin < .25:
      x = 0
      y = int(height * random.random())
    elif posCoin < .5:
      x = width - 1
      y = int(height * random.random())
    elif posCoin < .75:
      x = int(width * random.random())
      y = 0
    else:
      x = int(width * random.random())
      y = height - 1
    queries.append((x, y))
  
  for _ in xrange(rewardCandNum):
    reward = util.Counter()
    locs = queries[:]
    random.shuffle(locs)

    reward[locs[0]] = 5
    for idx in xrange(1, objNumPerFeat):
      reward[locs[idx]] = 0.1
    rewards.append(reward)
    
  def rewardGen(rewards): 
    def rewardFunc(s):
      if s in rewards.keys():
        return rewards[s]
      elif obstacleEnabled and s[0] == width / 2 and s[1] != height / 2:
        return -50
      else:
        return 0
    return rewardFunc
  terminalReward = util.Counter()
  terminalReward[(width / 2, height / 2)] = 1000

  def relevance(fState, query):
    withinReach = abs(fState[0] - query[0]) + abs(fState[1] - query[1])\
                + abs(query[0] - width / 2) + abs(query[1] - height / 2) < horizon - responseTime
    if obstacleEnabled:
      onSameSide = fState[0] < width / 2 and query[0] < width / 2\
                or fState[0] > width / 2 and query[0] > width / 2
    else:
      onSameSide = True
    return withinReach and onSameSide

  rewardSet = [rewardGen(reward) for reward in rewards]
  initialPhi = [1.0 / rewardCandNum] * rewardCandNum
  # ask whether reward is good or not
  if agentName == 'NQ':
    queries = [0] # make a dummy query set
    queryType = QueryType.NONE
  else:
    queryType = QueryType.REWARD

  # the true reward function is chosen randomly
  trueReward = random.choice(rewardSet)
  if agentName == 'WAIT':
    # the agent sleeps over and when it's awake, it finds itself did nothing and now it's response time!
    cmp = RockSample(queries, trueReward, gamma, 0, width, height,\
                     horizon = horizon - responseTime, terminalReward = terminalReward)
  else:
    cmp = RockSample(queries, trueReward, gamma, responseTime, width, height,\
                     horizon = horizon, terminalReward = terminalReward)
  cmp.setPossibleRewardValues([0, 0.1, 5])
  if agentName == 'JQTP' or agentName == 'NQ' or agentName == 'WAIT':
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
  elif agentName == 'KNOWN':
    agent = JointQTPAgent(cmp, [trueReward], [1], queryType, gamma)
  else:
    raise Exception("Unknown Agent " + agentName)

  if agentName == 'WAIT':
    # only simulate the episodes after the response
    ret, qValue, time = Experiment(cmp, agent, gamma, rewardSet, queryType, horizon=horizon- responseTime)
    ret = ret * gamma ** responseTime
    qValue = ret * gamma ** responseTime
  else:
    ret, qValue, time = Experiment(cmp, agent, gamma, rewardSet, queryType, horizon=horizon)

  print ret
  print qValue
  print time

if __name__ == '__main__':
  main()
