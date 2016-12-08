from QTPAgent import JointQTPAgent, OptimalPolicyQueryAgent
from policyGradientAgents import PolicyGradientQueryAgent,\
  PolicyGradientRandQueryAgent
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
from tabularNavigation import ThreeStateToy

flags = "r:l:s:d:a:vq:P:t:k:n:y:x"

"""
For all experiments that we need function approximation, now including mountain car and some toy domains.

We have arguments for policy gradient algorithms, and some candidate feature extractors.
"""
def experiment(cmp, feat, featLength, rewardSet, initialPhi):
  # discount factor
  gamma = 1
  agentName = 'MILP-POLICY'
  
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
    elif opt == '-x':
      config.DEBUG = True
    
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
  elif agentName == "MILP-POLICY":
    queryType = QueryType.POLICY
    agent = PolicyGradientQueryAgent(cmp, rewardSet, initialPhi, queryType, feat, featLength, gamma)
  elif agentName == "RAND-POLICY":
    queryType = QueryType.POLICY
    agent = PolicyGradientRandQueryAgent(cmp, rewardSet, initialPhi, queryType, feat, featLength, gamma)
  elif agentName == "MILP-SIMILAR":
    queryType = QueryType.SIMILAR
    agent = MILPTrajAgent(cmp, rewardSet, initialPhi, queryType, feat, featLength, gamma)
  elif agentName == "SIMILAR-DISAGREE":
    queryType = QueryType.SIMILAR
    agent = DisagreeTrajAgent(cmp, rewardSet, initialPhi, queryType, feat, featLength, gamma)
  elif agentName == "SIMILAR-VARIATION":
    queryType = QueryType.SIMILAR
    agent = BeliefChangeTrajAgent(cmp, rewardSet, initialPhi, queryType, feat, featLength, gamma)
  elif agentName == "SIMILAR-RANDOM":
    queryType = QueryType.SIMILAR
    agent = RandomTrajAgent(cmp, rewardSet, initialPhi, queryType, gamma)
  else:
    raise Exception("Unknown Agent " + agentName)

  if config.VERBOSE:
    print "Query type:", queryType

  ret, qValue, time = CMPExp.experiment(cmp, agent, gamma, rewardSet, queryType, horizon=cmp.horizon)

  print ret, qValue, time

  if config.PRINT == 'perf':
    f = open(agentName + str(config.opts) + '.out',"a")
    f.write(str(qValue) + ' ' + str(time) + '\n')
    f.close()

 
def mountainCarExp(Domain):
  def feat((x, v), a):
    """
    the length of feat is vecLen * |A|
    features for each action occupy different subsets of the feature vector
    """
    #vec = (x, v, x**2, x**3, x * v, x**2 * v, x**3 * v, v**2, 1)
    vec = (x, v, x**2, v**2, x * v, 1)
    #vec = (x,)
    
    vecLen = len(vec)

    if a == -1:
      return numpy.array(vec + (0,) * 2 * vecLen)
    elif a == 0:
      return numpy.array((0,) * vecLen + vec + (0,) * vecLen)
    elif a == 1:
      return numpy.array((0,) * 2 * vecLen + vec)
    else:
      raise Exception('unknown action. different domain?')

  featLength = len(feat((0, 0), 1)) # just use an arbitrary state to compute the length of feature
  
  horizon = 30

  def makeReward(loc0, loc1, reward):
    def r(s, a):
      # rewards are given on states. action a is dummy here
      if s[0] > loc0 and s[0] < loc1:
        return reward
      else:
        return -0.1
    return r

  rewardSet = [makeReward(i, i + 0.1, 10 if i == -1.1 or i == 1.0 else 1) for i in [-1.1, -0.6, 0.5, 1.0]]
  rewardCandNum = len(rewardSet)

  initialPhi = [1.0 / rewardCandNum] * rewardCandNum
  
  terminalReward = util.Counter()

  cmp = Domain(0, horizon, terminalReward)
  
  experiment(cmp, feat, featLength, rewardSet, initialPhi)


def threeStateExp():
  featLength = 3
  
  horizon = 1
  
  rewardSet = [lambda s, a: 1 if a == (-1, 0) else (.6 if a == (0, 1) else 0),\
               lambda s, a: 1 if a == (1, 0) else (.6 if a == (0, 1) else 0)]
  rewardCandNum = len(rewardSet)

  initialPhi = [1.0 / rewardCandNum] * rewardCandNum
  
  terminalReward = util.Counter()

  Domain = ThreeStateToy

  cmp = Domain(0, horizon, terminalReward)
  
  experiment(cmp, None, featLength, rewardSet, initialPhi)


if __name__ == '__main__':
  mountainCarExp(MountainCar)
  #mountainCarExp(MountainCarToy)
  #threeStateExp()
