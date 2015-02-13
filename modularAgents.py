from qlearningAgents import ApproximateQAgent
import featureExtractors

import math
import numpy as np
import numpy.linalg

class ModularAgent(ApproximateQAgent):
  """
  State: location of the agent.
  Action: weights on the sub-MDPs.
  Transition: transition of the agent.
  Reward: reward from the environment.

  Assume:
  Weights are independent from the location of the agent.
  """
  def __init__(self, **args):
    ApproximateQAgent.__init__(self, **args)

    # assume the weights are not dynamically learned, intialize them here.
    self.weights = [0, 0, 1]
    self.learningWeights = False
 
  def getQValue(self, state, action):
    """
    Get Q value by consulting each module
    """
    # sum over q values from each sub mdp
    return sum([self.qFuncs[i](state, action) * self.weights[i] for i in xrange(len(self.qFuncs)) if self.qFuncs[i](state, action) != None])

  def getSubQValues(self, state, action):
    return [self.qFuncs[i](state, action) for i in xrange(len(self.qFuncs))]

  def getSoftmaxQValue(self, state, action):
    actions = self.getLegalActions(state)
    j = actions.index(action)

    # q value matrix
    # vMat(i, j) is for i-th module and j-th action
    vMat = []

    # iterate through all the modules
    for qFunc in self.qFuncs:
      # list of exp^q
      exps = [math.exp(qFunc(state, action)) for action in actions]

      # Normalize
      sumExps = sum(exps)
      vMat.append([exp / sumExps for exp in exps])

    # sum over j-th column (j-th action)
    return sum([vMat[i][j] for i in range(len(self.qFuncs))])

  def setQFuncs(self, qFuncs):
    """
    Set QFuncs from the environment. getQValue will use this.
    """
    self.qFuncs = qFuncs

  def setWeights(self, weights):
    """
    If this is set, then the agent simply follows the weights.
    """
    self.learningWeights = False
    self.weights = weights

  def getWeights(self):
    """
    Get weights.
    """
    return self.weights
  
  def update(self, state, action, nextState, reward):
    if not self.learningWeights:
      pass
    else:
      # TODO
      raise Exception("calling unimplemented method: update for learning weight.")

  def getSignificance(self, state):
    """
    How significance an agent's correct decision at this state should affect the overall performance.

    Using the std of the Q values of this state.

    DUMMY
    """
    actions = self.getLegalActions(state)

    values = [self.getQValue(state, action) for action in actions]
    return np.std(values)


class ReducedModularAgent(ModularAgent):
  """
  Modular agent, which first maps state to an inner representation (so the state space reduced).
  """
  def __init__(self, **args):
    ModularAgent.__init__(self, **args)

    # Set get state function here.
    # By default, it is an identity function
    self.getState = lambda x : x

  def setStateFilter(self, extractor):
    """
    Set the state filter here, which returns the state representation for learning.
    The default one is an identity function.
    """
    self.getState = extractor

  def getAction(self, state):
    return ModularAgent.getAction(self, self.getState(state))

  def update(self, state, action, nextState, reward):
    return ModularAgent.update(self, self.getState(state), action, self.getState(nextState), reward)

  def final(self, state):
    return ModularAgent.final(self, self.getState(state))


def getContinuousWorldFuncs(mdp, Extractor = featureExtractors.ContinousRadiusLogExtractor):
  """
  DUMMY these are q functions to test agent's behavior in basic continuous domain.
  Values are given heuristically.
  """
  target = {'bias': 1, 'dist': -0.16}
  obstacle = {'bias': -1, 'dist': 0.16}
  segment = {'bias': 0.1, 'dist': -0.05}
  
  def radiusBias(state, action, label, w):
    extractor = Extractor(mdp, label)
    feats = extractor.getFeatures(state, action)

    if feats != None:
      return feats['bias'] * w['bias'] + feats['dist'] * w['dist']
    else:
      return None

  def qTarget(state, action):
    return radiusBias(state, action, 'targs', target)

  def qObstacle(state, action):
    return radiusBias(state, action, 'obsts', obstacle)

  def qSegment(state, action):
    return radiusBias(state, action, 'segs', segment)

  return [qTarget, qObstacle, qSegment]


def getHumanWorldDiscreteFuncs():
  """
  Use to get q functions.
  Note that the blief states are provided here - ((targDist[i], targAngle[i]), (objDist[i], objAngle[i]))
  These are further mapped to be bins.
  """
  import pickle

  # these are util.Counter objects
  tValues = pickle.load(open('learnedValues/humanAgenttargsValues.pkl'))
  oValues = pickle.load(open('learnedValues/humanAgentobstsValues.pkl'))
  sValues = tValues # suppose same as target module

  def qTarget(state, action):
    if not (state, action) in tValues.keys():
      raise Exception('Un-learned target ' + str(state) + ' ' + action)
    return tValues[state, action]

  def qObstacle(state, action):
    if not (state, action) in oValues.keys():
      raise Exception('Un-learned obstacle ' + str(state) + ' ' + action)
    return oValues[state, action]

  def qSegment(state, action):
    if not (state, action) in tValues.keys():
      raise Exception('Un-learned target ' + str(state) + ' ' + action)
    return tValues[state, action]

  # decouple the state representation, and call corresponding q functions
  """
  return [lambda s, a: qTarget(s[0], a) + qTarget(s[1], a), # closest targets
          lambda s, a: qObstacle(s[2], a) + qObstacle(s[3], a), # closest obstacles
          lambda s, a: qSegment(s[4], a)]
  """
  return [lambda s, a: qTarget(s[0], a), # closest targets
          lambda s, a: qObstacle(s[2], a), # closest obstacles
          lambda s, a: qSegment(s[4], a)]


def getHumanWorldContinuousFuncs():
  """
  Q value is a continuous approximator of the features.
  The state given here is belief state. Need use featureExtractor if having only raw states.

  weights: Action x Feature -> Value
  """
  import pickle

  tWeights = pickle.load(open('learnedValues/humanAgenttargsWeights.pkl'))
  oWeights = pickle.load(open('learnedValues/humanAgentobstsWeights.pkl'))
  #sWeights = pickle.load(open('learnedValues/humanAgentsegsWeights.pkl'))
  sWeights = tWeights # assume same behavior as target collection

  def qValue(state, action, weights):
    dist, angle = state
    assert action in weights.keys()
    w = weights[action]
    return w['bias'] + dist * w['dist'] + angle * w['angle'] + angle ** 2 * w['angleSq']

  def qTarget(state, action):
    targState, obstState, segState = state
    return qValue(targState, action, tWeights)

  def qObstacle(state, action):
    targState, obstState, segState = state
    return qValue(obstState, action, oWeights)

  def qSegment(state, action):
    targState, obstState, segState = state
    return qValue(segState, action, sWeights)

  return [qTarget, qObstacle, qSegment]

