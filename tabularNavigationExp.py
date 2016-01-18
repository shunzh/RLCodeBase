from QTPAgent import AlternatingQTPAgent, JointQTPAgent, RandomQueryAgent,\
  PriorTPAgent
from CMPExp import Experiment
import util
import sys
import random
import getopt
import config
from cmp import QueryType

"""
Algorithms:
- E-JQTP
- AQTP
- AQTP-NF
- opt query
- random query
- no query

Setting:
- response time (axis)

Show:
- expectation Q
- computation time
- paired difference between JQTP and AQTP
"""
def experiment(Domain, width, height, responseTime, horizon, rewardCandNum, rocks, terminalReward):
  # discount factor
  gamma = 0.9
  obstacleEnabled = False
  queryFlag = 'default'
  agentName = 'JQTP'
  
  try:
    opts, args = getopt.getopt(sys.argv[1:], "r:l:s:d:a:ovq:P:")
  except getopt.GetoptError:
    print 'unknown flag encountered'
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-r':
      random.seed(int(arg))
    elif opt == '-l':
      responseTime = int(arg)
    elif opt == '-s':
      config.PARAMETER = int(arg)
    elif opt == '-d':
      gamma = float(arg)
    elif opt == '-a':
      agentName = arg
    elif opt == '-o':
      obstacleEnabled = True
    elif opt == '-v':
      config.VERBOSE = True
    elif opt == '-P':
      config.PRINT = arg
    elif opt == '-q':
      queryFlag = arg
    
  rewards = []
  for _ in xrange(rewardCandNum):
    reward = util.Counter()
    reward[rocks[_]] = 1
    rewards.append(reward)
    
  def rewardGen(rewards): 
    def rewardFunc(s):
      if s in rewards.keys():
        return rewards[s]
      elif obstacleEnabled and s[0] == width / 2 and s[1] != height / 2:
        return -10
      else:
        return 0
    return rewardFunc
  """
  def relevance(fState, query):
    withinReach = abs(fState[0] - query[0]) + abs(fState[1] - query[1])\
                + abs(query[0] - width / 2) + abs(query[1] - height / 2) <= horizon - responseTime
    if obstacleEnabled:
      onSameSide = fState[0] <= width / 2 and query[0] <= width / 2\
                or fState[0] >= width / 2 and query[0] >= width / 2
    else:
      onSameSide = True
    return withinReach and onSameSide
  """

  rewardSet = [rewardGen(reward) for reward in rewards]
  initialPhi = []
  for _ in xrange(rewardCandNum):
    initialPhi.append(random.random())
  initialPhi = map(lambda _: _ / sum(initialPhi), initialPhi)

  # ask whether reward is good or not
  if agentName == 'NQ':
    queries = [0] # make a dummy query set
    queryType = QueryType.NONE
  else:
    if queryFlag == 'default':
      # use potential reward locations as query set
      queries = rocks
      queryType = QueryType.REWARD_SIGN
    elif queryFlag == 'test':
      queries = [(x, y) for x in range(0, width, width / 2) for y in range(0, height, height / 2)]
      queryType = QueryType.POLICY
    elif queryFlag == 'full':
      queries = [(x, y) for x in xrange(width) for y in xrange(height)]
      queryType = QueryType.POLICY
    else:
      raise Exception('unknown query flag ' + queryFlag)
  if config.VERBOSE:
    print "Query type:", queryType

  # the true reward function is chosen according to initialPhi
  trueReward = util.sample(initialPhi, rewardSet)
  if config.VERBOSE:
    print 'true reward', rewardSet.index(trueReward)
  cmp = Domain(queries, trueReward, gamma, responseTime, width, height,\
               horizon = horizon, terminalReward = terminalReward)
  cmp.setPossibleRewardValues([0, 1])
  cmp.possibleRewardLocations = rocks

  if agentName == 'JQTP' or agentName == 'NQ' or agentName == 'WAIT':
    agent = JointQTPAgent(cmp, rewardSet, initialPhi, queryType, gamma)
  elif agentName == 'AQTP':
    agent = AlternatingQTPAgent(cmp, rewardSet, initialPhi, queryType, gamma)
  elif agentName == 'AQTP-NF':
    # don't filter query. Assume all queries are relevant.
    agent = AlternatingQTPAgent(cmp, rewardSet, initialPhi, queryType, gamma, lambda fS, q: True)
  elif agentName == 'AQTP-RS':
    agent = AlternatingQTPAgent(cmp, rewardSet, initialPhi, queryType, gamma, restarts=1)
  elif agentName == 'RQ':
    agent = RandomQueryAgent(cmp, rewardSet, initialPhi, queryType, gamma)
  elif agentName == 'PTP':
    agent = PriorTPAgent(cmp, rewardSet, initialPhi, queryType, gamma)
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

  if config.PRINT == 'perf':
    print ret
    print qValue - 100 * gamma ** horizon
    print time

