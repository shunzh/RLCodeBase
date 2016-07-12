from QTPAgent import AlternatingQTPAgent, JointQTPAgent, RandomQueryAgent,\
  PriorTPAgent, HeuristicAgent, ActiveSamplingAgent, MILPAgent
import CMPExp
import util
import sys
import random
import getopt
import config
from cmp import QueryType

flags = "r:l:s:d:a:ovq:P:t:m:"

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
def experiment(Domain, width, height, responseTime, horizon, rewardCandNum, rewardSet, initialPhi, terminalReward):
  # discount factor
  gamma = 1
  queryFlag = 'default'
  agentName = 'JQTP'
  
  try:
    opts, args = getopt.getopt(sys.argv[1:], flags)
  except getopt.GetoptError:
    print 'unknown flag encountered'
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-r':
      random.seed(int(arg))
    elif opt == '-l':
      responseTime = int(arg)
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
    elif opt == '-m':
      config.para = int(arg)
    
  # ask whether reward is good or not
  if agentName == 'NQ':
    queries = [0] # make a dummy query set
    queryType = QueryType.NONE
  else:
    queries = [(x, y) for x in xrange(width) for y in xrange(height)]
    if queryFlag == 'default':
      # use potential reward locations as query set
      queryType = QueryType.ACTION
    elif queryFlag == 'reward':
      queryType = QueryType.REWARD_SIGN
    elif queryFlag == 'prefer':
      queryType = QueryType.PREFERENCE
    elif queryFlag == 'policy':
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

  if agentName == 'JQTP' or agentName == 'NQ' or agentName == 'WAIT':
    agent = JointQTPAgent(cmp, rewardSet, initialPhi, queryType, gamma)
  elif agentName == 'AQTP':
    # filter queries if same rewards
    agent = AlternatingQTPAgent(cmp, rewardSet, initialPhi, queryType, gamma)
  elif agentName == 'AQTP-P':
    # filter queries if same policies
    agent = AlternatingQTPAgent(cmp, rewardSet, initialPhi, queryType, gamma, relevance='ipp')
  elif agentName == 'AQTP-NF':
    # don't filter query. Assume all queries are relevant.
    agent = AlternatingQTPAgent(cmp, rewardSet, initialPhi, queryType, gamma, relevance=lambda fS, q: True)
  elif agentName == 'AQTP-RS':
    agent = AlternatingQTPAgent(cmp, rewardSet, initialPhi, queryType, gamma, restarts=1)
  elif agentName == 'RQ':
    agent = RandomQueryAgent(cmp, rewardSet, initialPhi, queryType, gamma)
  elif agentName == 'PTP':
    agent = PriorTPAgent(cmp, rewardSet, initialPhi, queryType, gamma)
  elif agentName == 'KNOWN':
    agent = JointQTPAgent(cmp, [trueReward], [1], queryType, gamma)
  elif agentName == "H":
    agent = HeuristicAgent(cmp, rewardSet, initialPhi, queryType, gamma)
  elif agentName == "AS":
    agent = ActiveSamplingAgent(cmp, rewardSet, initialPhi, queryType, gamma)
  elif agentName == "MILP":
    agent = MILPAgent(cmp, rewardSet, initialPhi, queryType, gamma, qi=False)
  else:
    raise Exception("Unknown Agent " + agentName)

  if agentName == 'WAIT':
    # only simulate the episodes after the response
    ret, qValue, time = CMPExp.experiment(cmp, agent, gamma, rewardSet, queryType, horizon=horizon- responseTime)
    ret = ret * gamma ** responseTime
    qValue = ret * gamma ** responseTime
  else:
    ret, qValue, time = CMPExp.experiment(cmp, agent, gamma, rewardSet, queryType, horizon=horizon)

  if config.PRINT == 'perf':
    print ret, qValue, time

    f = open(agentName + str(config.para) + '.out',"a")
    f.write(str(qValue) + ' ' + str(time) + '\n')
    f.close()
