"""
Q functions that used for continuous world and human world domains.

These functions return the q functions for different scenarios.
"""

import featureExtractors
import numpy as np
from humanWorld import HumanWorld
import warnings

"""
Human world functions
"""
def getHumanWorldDiscreteFuncs():
  """
  Use to get q functions.
  Note that the blief states are provided here - ((dist, angle)*)
  These are further mapped to be bins.
  """
  import pickle

  # these are util.Counter objects
  tValues = pickle.load(open('learnedValues/humanAgenttargsValues.pkl'))
  oValues = pickle.load(open('learnedValues/humanAgentobstsValues.pkl'))
  sValues = pickle.load(open('learnedValues/humanAgentsegsValues.pkl'))
  #sValues = tValues # FOR DEBUGING PATH MODULE

  def qTarget(s, a):
    state, action = featureExtractors.discreteQTableCompressor(s, a)
    if not (state, action) in tValues.keys():
      warnings.warn('Un-learned target')
    return tValues[state, action]

  def qObstacle(s, a):
    state, action = featureExtractors.discreteQTableCompressor(s, a)
    if not (state, action) in oValues.keys():
      warnings.warn('Un-learned obstacle')
    return oValues[state, action]

  def qSegment(s, a):
    state, action = featureExtractors.discreteQTableCompressor(s, a)
    if not (state, action) in sValues.keys():
      warnings.warn('Un-learned segment ')
    return sValues[state, action]

  # discounter is dropped for these q functions. No way to use different discounters.
  return [lambda s, a, d = None: qTarget(s[0], a), # closest targets
          lambda s, a, d = None: qObstacle(s[2], a), # closest obstacles
          lambda s, a, d = None: qSegment(s[4], a)]

def getHumanWorldQPotentialFuncs():
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

  def vFunc(s, para):
    """
      _
    _/ \_ <- v function looks like this

    """
    reward, discounter, radius = para
    dist, orient = s
    if dist == None or orient == None:
      # in which case no such object left in the domain
      return 0
    elif dist < radius:
      return reward
    else:
      return reward * np.power(discounter, dist - radius)
  
  def qPath(state, currentState, action, reward, discounter, radius):
    s = transition(state, action)
    curS = transition(currentState, action)
    project = featureExtractors.getProjectionToSegmentLocalView(s, curS)
    return vFunc(project, reward, discounter, radius)

  return [lambda s, a, x: vFunc(transition(s[0], a), [x[0], x[3], x[6]]), # closest target(s)
          lambda s, a, x: vFunc(transition(s[2], a), [x[1], x[4], x[7]]), # closest obstacle(s)
          lambda s, a, x: vFunc(transition(s[4], a), [x[2], x[5], x[8]])] # next seg point
          #lambda s, a, d: qPath(s[4], s[5], a, d[3])] # closest path (next two seg points)

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
  Return Q functions for modular mdp for target collection / obstacle avoidance behavior
  in gridworld domains.
  """
  def getNext(state, action):
    # simulate the next state
    x, y = state
    dx, dy = Actions.directionToVector(action)
    next_x, next_y = int(x + dx), int(y + dy)
    if next_x < 0 or next_x >= mdp.grid.width:
      next_x = x
    if next_y < 0 or next_y >= mdp.grid.height:
      next_y = y

    return [next_x, next_y]

  def qFuncGen(r, idx):
    """
    Compute a Q value responding to an object, considering the distance to it.
    This is used by obstacle avoidance, and target obtaining.

    Args:
      r: the reward of the module class to be found
      idx: the id of the class
    """
    def qModule(state, action, discounters):
      if action == "exit": return 0
      
      next_x, next_y = getNext(state, action)

      # find the distance to the nearest object
      minDist = mdp.grid.width * mdp.grid.height
      for xt in range(mdp.grid.width):
        for yt in range(mdp.grid.height):
          cell = mdp.grid[xt][yt] 
          if cell == r:
            dist = math.sqrt((xt - next_x) ** 2 + (yt - next_y) ** 2)
            if (dist < minDist): minDist = dist

      return r / abs(r) * discounters[idx] ** minDist
    return qModule
    
  rewards = [r for r, c in mdp.spec]
  return [qFuncGen(r, idx) for r, idx in zip(rewards, range(len(rewards)))]