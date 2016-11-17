from QTPAgent import AlternatingQTPAgent, JointQTPAgent, RandomQueryAgent,\
  PriorTPAgent, MILPAgent, OptimalPolicyQueryAgent, MILPDemoAgent, OptimalPartialPolicyQueryAgent,\
  PolicyGradientAgent, PolicyGradientQueryAgent
from actionQueryAgents import MILPActionAgent
from trajAgents import BeliefChangeTrajAgent, RandomTrajAgent, DisagreeTrajAgent,\
  MILPTrajAgent
import CMPExp
import util
import sys
import random
import getopt
import config
from cmp import QueryType
from mountainCar import MountainCar, MountainCarToy
import numpy

flags = "r:l:s:d:a:vq:P:t:k:n:y:"

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
def experiment(cmp, feat, featLength, rewardSet, initialPhi):
  # discount factor
  gamma = 1
  responseTime = 0
  agentName = 'JQTP'
  
  try:
    opts, args = getopt.getopt(sys.argv[1:], flags)
  except getopt.GetoptError:
    print 'unknown flag encountered'
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-r':
      random.seed(int(arg))
    elif opt == '-a':
      agentName = arg
    elif opt == '-v':
      config.VERBOSE = True
    
  # the true reward function is chosen according to initialPhi
  trueRewardIdx = util.sample(initialPhi, range(len(rewardSet)))
  if config.VERBOSE:
    print 'true reward', trueRewardIdx

  # continue initializing the cmp object
  cmp.decorate(gamma, None, trueRewardIdx, rewardSet[trueRewardIdx])
  cmp.setPossibleRewardValues([0, 1])

  if agentName == 'JQTP' or agentName == 'NQ' or agentName == 'WAIT':
    queryType = QueryType.ACTION
    agent = JointQTPAgent(cmp, rewardSet, initialPhi, queryType, gamma)
  elif agentName == "MILP":
    agent = MILPActionAgent(cmp, rewardSet, initialPhi, queryType, gamma)
  elif agentName == "OPT-POLICY":
    queryType = QueryType.POLICY
    agent = OptimalPolicyQueryAgent(cmp, rewardSet, initialPhi, queryType, gamma)
  elif agentName == "MILP-SIMILAR":
    queryType = QueryType.SIMILAR
    agent = PolicyGradientQueryAgent(cmp, rewardSet, initialPhi, queryType, feat, featLength, 0.01, gamma)
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
    ret, qValue, time = CMPExp.experiment(cmp, agent, gamma, rewardSet, queryType, horizon=cmp.horizon - responseTime)
    ret = ret * gamma ** responseTime
    qValue = ret * gamma ** responseTime
  else:
    ret, qValue, time = CMPExp.experiment(cmp, agent, gamma, rewardSet, queryType, horizon=cmp.horizon)

  print ret, qValue, time

  if config.PRINT == 'perf':
    f = open(agentName + str(config.opts) + '.out',"a")
    f.write(str(qValue) + ' ' + str(time) + '\n')
    f.close()

if __name__ == '__main__':
  """
  def feat((x, v), a):
    v += a
    x += v
    #return numpy.array((x, v, x**2, v**2, x * v, 1))
    return numpy.array((x, v, x**2, x**3, x * v, x**2 * v, x**3 * v, v**2, 1))
  """
  def feat((x, x0), v):
    x += v
    return numpy.array((x,))

  featLength = len(feat((0, 0), 1)) # just use an arbitrary state to compute the length of feature
  
  horizon = 30 # note that this can't be inf.. the agent may not terminate
  # define feature-based reward functions
  def makeReward(phiRanges, action, value):
    def r(s, a):
      if s[0] >= phiRanges[0] and s[0] <= phiRanges[1] and a == action:
        return value
      else:
        return -0.1
    return r

  rewardSet = [makeReward([2.1, 3], 1, 1),\
               makeReward([-3, -2.1], -1, 1),\
              ]

  rewardCandNum = len(rewardSet)

  initialPhi = [1.0 / rewardCandNum] * rewardCandNum
  
  terminalReward = util.Counter()

  #Domain = MountainCar
  Domain = MountainCarToy

  cmp = Domain(0, horizon, terminalReward)
  
  experiment(cmp, feat, featLength, rewardSet, initialPhi)