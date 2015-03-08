"""
Q functions that used for continuous world and human world domains.

These functions return the q functions for different scenarios.
"""

import featureExtractors
import numpy as np
from humanWorld import HumanWorld
import warnings

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
  sValues = pickle.load(open('learnedValues/humanAgentsegsValues.pkl'))

  def qTarget(state, action):
    if not (state, action) in tValues.keys():
      warnings.warn('Un-learned target ' + str(state) + ' ' + action)
    return tValues[state, action]

  def qObstacle(state, action):
    if not (state, action) in oValues.keys():
      warnings.warn('Un-learned obstacle ' + str(state) + ' ' + action)
    return oValues[state, action]

  def qSegment(state, action):
    if not (state, action) in sValues.keys():
      warnings.warn('Un-learned segment ' + str(state) + ' ' + action)
    return sValues[state, action]

  # discounter is dropped for these q functions. No way to use different discounters.
  return [lambda s, a, d = None: qTarget(s[0], a), # closest targets
          lambda s, a, d = None: qObstacle(s[2], a), # closest obstacles
          lambda s, a, d = None: qSegment(s[4], a)]

def getHumanWorldQPotentialFuncs(defaultD = [0.6] * 3):
  """
  Rather learned from samples, we define the potential functions (a value function) based on reward.
  Q functions here just reflect the potential functions.
  Simulate the dynamics dependent on distance, angle to an object, and action taken.
  
  Args:
    defaultD: default discounters.
              It's true that discounter should not be part of the agent.
              We may compute the q value for different discounters. So we need a discounter parameter.
  """
  transition = HumanWorld.transitionSimulate

  def vTarget(s, discounter):
    dist, orient = s
    return 1 * np.power(discounter, dist)
  
  def vObstacle(s, discounter):
    dist, orient = s
    return -1 * np.power(discounter, dist)

  def vSegment(s, discounter):
    dist, orient = s
    return 1 * np.power(discounter, dist)

  def qTarget(state, action, discounter):
    return vTarget(transition(state, action), discounter)

  def qObstacle(state, action, discounter):
    return vObstacle(transition(state, action), discounter)

  def qSegment(state, action, discounter):
    return vSegment(transition(state, action), discounter)

  return [lambda s, a, d = defaultD: qTarget(s[0], a, d[0]), # closest targets
          lambda s, a, d = defaultD: qObstacle(s[2], a, d[1]), # closest obstacles
          lambda s, a, d = defaultD: qSegment(s[4], a, d[2])]

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