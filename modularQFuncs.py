"""
Q functions that used for continuous world and human world domains.

These functions return the q functions for different scenarios.
"""

import featureExtractors
import numpy as np
from humanWorld import HumanWorld

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
  sValues = None # FIXME not using an RL module

  def qTarget(state, action):
    if not (state, action) in tValues.keys():
      raise Exception('Un-learned target ' + str(state) + ' ' + action)
    return tValues[state, action]

  def qObstacle(state, action):
    if not (state, action) in oValues.keys():
      raise Exception('Un-learned obstacle ' + str(state) + ' ' + action)
    return oValues[state, action]

  def qSegment(state, action):
    bigQ = 0.2
    smallQ = 0.1

    # hand-made path following
    if abs(state[1]) == 0 and action == 'G':
      # good orientation, good action
      return bigQ
    elif abs(state[1]) == 0:
      # good orientation
      return bigQ
      return smallQ
    elif abs(state[1]) == 1 and action == 'G':
      # ok orientation, ok action
      return smallQ
    elif state[1] < 0 and action == 'L' or state[1] > 0 and action == 'R':
      # ok orientation, good action
      return bigQ
    else:
      return 0

  # decouple the state representation, and call corresponding q functions
  """
  return [lambda s, a: qTarget(s[0], a) + qTarget(s[1], a), # closest targets
          lambda s, a: qObstacle(s[2], a) + qObstacle(s[3], a), # closest obstacles
          lambda s, a: qSegment(s[4], a)]
  """
  return [lambda s, a: qTarget(s[0], a), # closest targets
          lambda s, a: qObstacle(s[2], a), # closest obstacles
          lambda s, a: qSegment(s[4], a)]

def getHumanWorldQPotentialFuncs():
  """
  Rather learned from samples, we define the potential functions (a value function) based on reward.
  Q functions here just reflect the potential functions.
  Simulate the dynamics dependent on distance, angle to an object, and action taken.
  """
  transition = HumanWorld.transitionSimulate

  def vTarget(s):
    dist, orient = s
    return 1 * np.power(0.8, dist)
  
  def vObstacle(s):
    dist, orient = s
    return -1 * np.power(0.8, dist)

  def vSegment(s):
    dist, orient = s
    return 1 * np.power(0.95, dist)

  def qTarget(state, action):
    return vTarget(transition(state, action))

  def qObstacle(state, action):
    return vObstacle(transition(state, action))

  def qSegment(state, action):
    return vSegment(transition(state, action))

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