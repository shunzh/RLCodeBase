"""
Q functions that used for continuous world and human world domains.

These functions return the q functions for different scenarios.
"""

import featureExtractors
import numpy as np
from humanWorld import HumanWorld
import warnings
from game import Actions
import math
import config

"""
Human world functions
"""
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
  #sValues = tValues # FOR DEBUGING PATH MODULE

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

def getHumanWorldQPotentialFuncs(defaultD = [0.6] * 3, twoObjects = config.TWO_OBJECTS):
  """
  Rather learned from samples, we define the potential functions (a value function) based on reward.
  Q functions here just reflect the potential functions.
  Simulate the dynamics dependent on distance, angle to an object, and action taken.
  
  Args:
    defaultD: default discounters.
              It's true that discounter should not be part of the agent.
              We may compute the q value for different discounters. So we need a discounter parameter.
    twoObjects: look at two closest objects.
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
  
  def vPath(s, curS, discounter):
    dist, orient = featureExtractors.getProjectionToSegmentLocalView(s, curS)
    return 1 * np.power(discounter, dist)

  def qTarget(state, action, discounter):
    return vTarget(transition(state, action), discounter)

  def qObstacle(state, action, discounter):
    return vObstacle(transition(state, action), discounter)

  def qSegment(state, action, discounter):
    return vSegment(transition(state, action), discounter)

  def qPath(state, currentState, action, discounter):
    return vPath(transition(state, action), transition(currentState, action), discounter)

  if twoObjects:
    return [lambda s, a, d = defaultD: qTarget(s[0], a, d[0]) + qTarget(s[1], a, d[0]), # closest target(s)
            lambda s, a, d = defaultD: qObstacle(s[2], a, d[1]) + qObstacle(s[3], a, d[1]), # closest obstacle(s)
            lambda s, a, d = defaultD: qSegment(s[4], a, d[2]) + qSegment(s[5], a, d[2])] # next seg point
            #lambda s, a, d = defaultD: qPath(s[4], s[5], a, d[3])] # closest path (next two seg points)
  else:
    return [lambda s, a, d = defaultD: qTarget(s[0], a, d[0]), # closest target(s)
            lambda s, a, d = defaultD: qObstacle(s[2], a, d[1]), # closest obstacle(s)
            lambda s, a, d = defaultD: qSegment(s[4], a, d[2])] # next seg point
            #lambda s, a, d = defaultD: qPath(s[4], s[5], a, d[3])] # closest path (next two seg points)

def getHumanWorldContinuousFuncs():
  """
  NOTE: this method doesn't work well for humanWorld.

  Assume the q values are linear combination of state features (distance, angle, etc.)
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

"""
Continuous world functions
"""
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

def getObsAvoidFuncs(mdp):
  """
  Return Q functions for modular mdp for obstacle avoidance behavior

  the environment is passed by mdp
  """
  obstacle = {'bias': -0.20931133310480204, 'dis': 0.06742681562641269}
  target = {'bias': 0.20931133310480204, 'dis': -0.06742681562641269}
  sidewalk = {'x': 0.06250000371801567}

  def getNext(state, action):
    x, y = state
    dx, dy = Actions.directionToVector(action)
    next_x, next_y = int(x + dx), int(y + dy)
    if next_x < 0 or next_x >= mdp.grid.width:
      next_x = x
    if next_y < 0 or next_y >= mdp.grid.height:
      next_y = y

    return [next_x, next_y]

  def qWalk(state, action):
    """
    QValue of forward walking
    """
    next_x, next_y = getNext(state, action)
    return sidewalk['x'] * next_x

  def radiusBias(state, action, cond, w):
    """
    Compute a Q value responding to an object, considering the distance to it.
    This is used by obstacle avoidance, and target obtaining.

    Args:
      state, action
      cond: the lambda expr that given state is the object we want
      w: weight vector
    """
    x, y = state
    next_x, next_y = getNext(state, action)

    # find the distance to the nearest object
    minDist = mdp.grid.width * mdp.grid.height
    for xt in range(mdp.grid.width):
      for yt in range(mdp.grid.height):
        cell = mdp.grid[xt][yt] 
        if cond(cell):
          # it's an obstacle!
          dist = math.sqrt((xt - next_x) ** 2 + (yt - next_y) ** 2)
          if (dist < minDist): minDist = dist
    return minDist * w['dis'] + 1 * w['bias']

  def qObstacle(state, action):
    cond = lambda s : (type(s) == int or type(s) == float) and s == -1
    return radiusBias(state, action, cond, obstacle)

  def qTarget(state, action):
    cond = lambda s : (type(s) == int or type(s) == float) and s == +1
    return radiusBias(state, action, cond, target)

  return [qWalk, qObstacle, qTarget]
