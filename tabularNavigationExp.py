from QTPAgent import AlternatingQTPAgent, JointQTPAgent, RandomQueryAgent,\
  PriorTPAgent, OptimalPolicyQueryAgent, AprilAgent 
from greedyConstructionAgents import MILPAgent, MILPDemoAgent
from actionQueryAgents import HeuristicAgent, ActiveSamplingAgent,\
  MILPActionAgent
from trajAgents import BeliefChangeTrajAgent, RandomTrajAgent, DisagreeTrajAgent,\
  MILPTrajAgent
import CMPExp
import util
import sys
import random
import getopt
import config
from cmp import QueryType
from featureBasedAgents import FeatureBasedPolicyQueryAgent

flags = "r:l:s:d:a:vq:P:t:k:n:y:"

def experiment(cmp, rewardSet, initialPhi):
  # discount factor
  gamma = 1
  responseTime = 0
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
    elif opt == '-v':
      config.VERBOSE = True
    elif opt == '-P':
      config.PRINT = arg
    elif opt == '-q':
      queryFlag = arg
    
  # ask whether reward is good or not
  if agentName == 'NQ':
    queries = [0] # make a dummy query set
    queryType = QueryType.NONE
  else:
    #queries = [(x, y) for x in xrange(width) for y in xrange(height)]
    # use the following code to deal with the problem of using action queries in k way
    queries = cmp.getStates()
    if queryFlag == 'default':
      # use potential reward locations as query set
      queryType = QueryType.ACTION
    elif queryFlag == 'reward':
      queryType = QueryType.REWARD_SIGN
    elif queryFlag == 'policy':
      queryType = QueryType.POLICY
    else:
      raise Exception('unknown query flag ' + queryFlag)

  # the true reward function is chosen according to initialPhi
  trueRewardIdx = util.sample(initialPhi, range(len(rewardSet)))
  if config.VERBOSE:
    print 'true reward', trueRewardIdx

  # continue initializing the cmp object
  cmp.decorate(gamma, queries)
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
    agent = JointQTPAgent(cmp, [rewardSet[trueRewardIdx]], [1], queryType, gamma)
  elif agentName == "H":
    agent = HeuristicAgent(cmp, rewardSet, initialPhi, queryType, gamma)
  elif agentName == "AS":
    agent = ActiveSamplingAgent(cmp, rewardSet, initialPhi, queryType, gamma)
  elif agentName == "MILP":
    agent = MILPActionAgent(cmp, rewardSet, initialPhi, queryType, gamma)
  elif "APRIL" in agentName:
    queryType = QueryType.POLICY
    agent = AprilAgent(cmp, rewardSet, initialPhi, queryType, gamma)
    if '1' in agentName:
      config.SAMPLES_TIMES = 10
    elif '2' in agentName:
      config.SAMPLES_TIMES = 20
    elif '3' in agentName:
      config.SAMPLES_TIMES = 50
    else:
      raise "unknown april agent"
  elif agentName == "MILP-QI":
    agent = MILPAgent(cmp, rewardSet, initialPhi, queryType, gamma, qi=True)
  elif agentName == "FEAT-GREEDY":
    queryType = QueryType.POLICY
    agent = FeatureBasedPolicyQueryAgent(cmp, rewardSet, initialPhi, queryType, gamma)
  elif agentName == "MILP-POLICY":
    queryType = QueryType.POLICY
    agent = MILPAgent(cmp, rewardSet, initialPhi, queryType, gamma)
  elif agentName == "MILP-QI-POLICY":
    queryType = QueryType.POLICY
    agent = MILPAgent(cmp, rewardSet, initialPhi, queryType, gamma, qi=True)
  elif agentName == "OPT-POLICY":
    queryType = QueryType.POLICY
    agent = OptimalPolicyQueryAgent(cmp, rewardSet, initialPhi, queryType, gamma)
  elif agentName == "OPT-POLICY-ACT":
    queryType = QueryType.ACTION
    agent = OptimalPolicyQueryAgent(cmp, rewardSet, initialPhi, queryType, gamma)
  elif agentName == "MILP-DEMO-BATCH":
    # generate policy query first (so we call MILPAgent) and then sample trajectories
    queryType = QueryType.DEMONSTRATION
    agent = MILPAgent(cmp, rewardSet, initialPhi, queryType, gamma)
  elif agentName == "MILP-DEMO":
    # generate policy one at a time after observing the generated trajectories so far
    queryType = QueryType.DEMONSTRATION
    agent = MILPDemoAgent(cmp, rewardSet, initialPhi, queryType, gamma)
  elif agentName == "MILP-PARTIAL-POLICY":
    queryType = QueryType.PARTIAL_POLICY
    agent = MILPAgent(cmp, rewardSet, initialPhi, queryType, gamma)
  elif agentName == "MILP-SIMILAR":
    queryType = QueryType.SIMILAR
    agent = MILPTrajAgent(cmp, rewardSet, initialPhi, queryType, gamma)
  elif agentName == "SIMILAR-DISAGREE":
    queryType = QueryType.SIMILAR
    agent = DisagreeTrajAgent(cmp, rewardSet, initialPhi, queryType, gamma)
  elif agentName == "SIMILAR-VARIATION":
    queryType = QueryType.SIMILAR
    agent = BeliefChangeTrajAgent(cmp, rewardSet, initialPhi, queryType, gamma)
  elif agentName == "SIMILAR-RANDOM":
    queryType = QueryType.SIMILAR
    agent = RandomTrajAgent(cmp, rewardSet, initialPhi, queryType, gamma)
  else:
    raise Exception("Unknown Agent " + agentName)

  if config.VERBOSE:
    print "Query type:", queryType

  if agentName == 'WAIT':
    # only simulate the episodes after the response
    ret, qValue, time = CMPExp.experiment(cmp, agent, gamma, rewardSet, queryType, horizon=cmp.horizon- responseTime)
    ret = ret * gamma ** responseTime
    qValue = ret * gamma ** responseTime
  else:
    ret, qValue, time = CMPExp.experiment(cmp, agent, gamma, rewardSet, queryType, horizon=cmp.horizon)

  print ret, qValue, time

  if config.PRINT == 'perf':
    f = open(agentName + str(config.opts) + '.out',"a")
    f.write(str(qValue) + ' ' + str(time) + '\n')
    f.close()