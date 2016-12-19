from QTPAgent import JointQTPAgent, OptimalPolicyQueryAgent
from policyGradientAgents import PolicyGradientQueryAgent,\
  PolicyGradientRandQueryAgent, AprilAgent, SamplingAgent
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
from driving import Driving

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
      numpy.random.seed(int(arg))
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
  elif agentName == "SAMPLE-POLICY":
    queryType = QueryType.POLICY
    agent = SamplingAgent(cmp, rewardSet, initialPhi, queryType, feat, featLength, gamma)
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
    vec = (x, v, x**2, x**3, x * v, x**2 * v, x**3 * v, v**2, 1)
    #vec = (x, v, x**2, v**2, x * v, 1)
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
  
  horizon = 20

  def makeReward(loc0, loc1, reward):
    def r(s, a):
      # rewards are given on states. action a is dummy here
      if s[0] > loc0 and s[0] < loc1 and s[1] >= -0.02 and s[1] <= 0.02:
        return reward
      else:
        return 0
    return r

  rewardSet = [makeReward(i, i + 0.1, 10 if i == -1.0 or i == 0.9 else 1) for i in [-1.0, -0.7, -0.4, 0.3, 0.6, 0.9]]
  rewardCandNum = len(rewardSet)

  initialPhi = [1.0 / rewardCandNum] * rewardCandNum
  
  terminalReward = util.Counter()

  cmp = Domain(0, horizon, terminalReward)
  
  experiment(cmp, feat, featLength, rewardSet, initialPhi)


def drivingExp():
  numOfLanes = 3
  length = 5
  numOfCars = 3
  carLength = 0.5
  #cars = [(random.random() * 5, random.randint(0, numOfLanes - 1)) for _ in range(numOfCars)]
  cars = [(0.5, 1), (1.5, 0), (2.5, 2)]
  print cars

  def feat(s, a):
    # binary vector to indicate the current lane of the robot
    vec = [1 if s[1] == i else 0 for i in range(numOfLanes)]
    for i in [s[1] - 1, s[1], s[1] + 1]:
      laneFeat = [0, 0, 0]
      for car in cars:
        if car[1] == i and car[0] - s[0]:
          if car[0] - s[0] < carLength:
            laneFeat[0] = 1
          elif car[0] - s[0] < 1:
            laneFeat[1] = 1
          else:
            laneFeat[2] = 1
      vec = vec + laneFeat

    vec = tuple(vec)
    vecLen = len(vec)

    if a == 'W':
      return numpy.array(vec + (0,) * 2 * vecLen)
    elif a == 'E':
      return numpy.array((0,) * vecLen + vec + (0,) * vecLen)
    elif a == 'N':
      return numpy.array((0,) * 2 * vecLen + vec)
    else:
      raise Exception('unknown action ' + str(a) + '. different domain?')

  featLength = len(feat((0, 1), 'N')) # just use an arbitrary state to compute the length of feature

  horizon = 50
  terminalReward = util.Counter()
  cmp = Driving(0, horizon, terminalReward, length=length, lanes=numOfLanes)
  
  drivers = {
    'niceDriver': {'cars': -1, 'lane': None},
    'nastyDriver': {'cars': 1, 'lane': None},
    'dangerousDriver': {'cars': -1, 'backOfCars': 1, 'lane': None},
    'rightNiceDriver': {'cars': -1, 'lane': 0},
    #'rightNastyDriver': {'cars': 1, 'lane': 0},
    'leftNiceDriver': {'cars': -1, 'lane': numOfLanes - 1},
    #'leftNastyDriver': {'cars': 1, 'lane': numOfLanes - 1},
    #'middleNiceDriver': {'cars': -1, 'lane': numOfLanes / 2},
    #'middleNastyDriver': {'cars': 1, 'lane': numOfLanes / 2},
    #'leftLaneDriver': {'cars': 0, 'lane': 0},
    #'middleLaneDriver': {'cars': 0, 'lane': numOfLanes / 2},
    #'rightLaneDriver': {'cars': 0, 'lane': numOfLanes - 1},
  }

  def makeReward(spec):
    def r(s, a):
      reward = 0
      for car in cars:
        if car[1] == s[1] and car[0] > s[0] and car[0] < s[0] + carLength:
          reward += spec['cars']

      if 'backOfCars' in spec.keys():
        if car[1] == s[1] and car[0] < s[0] and car[0] > s[0] - carLength:
          reward += spec['backOfCars']

      if spec['lane']:
        if car[1] == spec['lane']:
          reward += 0.01
      
      return reward
    return r
  rewardSet = [makeReward(driverSpec) for driverName, driverSpec in drivers.items()]

  rewardCandNum = len(rewardSet)
  initialPhi = [1.0 / rewardCandNum] * rewardCandNum

  experiment(cmp, feat, featLength, rewardSet, initialPhi)


def threeStateExp():
  # a simple domain to make sure our pg works
  featLength = 3
  
  if config.POLICY_TYPE == 'linear':
    # direct policy control.
    # (normalized) theta1, theta2, theta3 are the prob. of reaching three terminal states
    # this is a very simple problem to solve. check this first when debugging!
    feat = None
  elif config.POLICY_TYPE == 'softmax':
    # theta1 .. theta3 are the values of states
    def feat(s, a):
      if not s == (0, 0): vec = (0, 0, 0)
      elif a == (1, 0): vec = (1, 0, 0)
      elif a == (-1, 0): vec = (0, 1, 0)
      elif a == (0, 1): vec =(0, 0, 1)
      else: raise Exception('should be unreachable')
      
      return numpy.array(vec)
  
  horizon = 1
  
  rewardSet = [lambda s, a: 1 if a == (-1, 0) else (.6 if a == (0, 1) else 0),\
               lambda s, a: 1 if a == (1, 0) else (.6 if a == (0, 1) else 0)]
  rewardCandNum = len(rewardSet)

  initialPhi = [1.0 / rewardCandNum] * rewardCandNum
  
  terminalReward = util.Counter()

  Domain = ThreeStateToy

  cmp = Domain(0, horizon, terminalReward)
  
  experiment(cmp, feat, featLength, rewardSet, initialPhi)


if __name__ == '__main__':
  #mountainCarExp(MountainCar)
  #mountainCarExp(MountainCarToy)
  threeStateExp()
  #drivingExp()
