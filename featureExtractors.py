# featureExtractors.py
# --------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util

import numpy.linalg
import numpy as np
import humanWorld
import warnings
from game import Actions

class FeatureExtractor:  
  def getFeatures(self, state, action):    
    """
      Returns a dict from features to counts
      Usually, the count will just be 1.0 for
      indicator functions.  
    """
    util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
  def getFeatures(self, state, action):
    feats = util.Counter()
    feats[(state,action)] = 1.0
    return feats

class ContinousRadiusLogExtractor(FeatureExtractor):
  """
  An feature extractor for the ContinuousWorld
  """
  def __init__(self, mdp, label):
    self.mdp = mdp
    self.label = label

  def getFeatures(self, state, action):
    feats = util.Counter()
    loc, orient = state
    newLoc, newOrient = self.mdp.getTransitionStatesAndProbs(state, action)[0][0]

    if self.label == 'segs':
      if len(self.mdp.objs['segs']) > 0:
        minObj = self.mdp.objs['segs'][0]
        minDist = numpy.linalg.norm(np.subtract(loc, minObj))
      else:
        minObj = loc; minDist = np.inf
    else:
      [minObj, minDist] = getClosestObj(newLoc, self.mdp.objs[self.label])

    if minDist == np.inf:
      return None
    else:
      feats['dist'] = np.log(1 + minDist)
      feats['bias'] = 1

    return feats

class HumanViewExtractor(ContinousRadiusLogExtractor):
  """
  Feature extractors for the HumanWorld
  """
  def __init__(self, mdp, label):
    ContinousRadiusLogExtractor.__init__(self, mdp, label)
    # enable then add squared term for angle

    # keep the label for convenience
    self.label = label

  def getFeatures(self, state, action):
    """
    This has to assume that the transition is known.
    """
    newState = self.mdp.getTransitionStatesAndProbs(state, action)[0][0]
    return self.getStateFeatures(newState)

  def getStateFeatures(self, state):
    feats = util.Counter()

    loc, orient = state

    if self.label == 'segs':
      # rubber band
      # get features for waypoints
      if len(self.mdp.objs['segs']) > 1:
        obj = self.mdp.objs['segs'][1] # look at the NEXT waypoint
        curObj = self.mdp.objs['segs'][0]
      elif len(self.mdp.objs['segs']) == 1:
        obj = curObj = self.mdp.objs['segs'][0] # this is the last segment
      else:
        obj = curObj = None

      feats['dist'], feats['angle'] = getDistAngle(loc, obj, orient)
      feats['curDist'], feats['curAngle'] = getDistAngle(loc, curObj, orient)
    else:
      # get features for targets / objects
      # get both closest and the second closest -- may not be both used though
      l = getSortedObjs(loc, self.mdp.objs[self.label])
      if len(l) > 0:
        minObj = l[0]
      else:
        minObj = None

      if len(l) > 1:
        # if there are more than two objects
        secMinObj = l[1]
      else:
        secMinObj = None

      feats['dist'], feats['angle'] = getDistAngle(loc, minObj, orient)
      feats['dist2'], feats['angle2'] = getDistAngle(loc, secMinObj, orient)

    feats['bias'] = 1

    return feats

def getHumanContinuousMapper(mdp):
  """
  Return ((targDist, targAngle)*2, (obstDist, obstAngle)*2, (segDist, segAngle)*2)
  """
  extractors = [HumanViewExtractor(mdp, label) for label in ['targs', 'obsts', 'segs']]

  def getDistAngelList(state, action):
    ret = []
    for extractor in extractors:
      feats = extractor.getStateFeatures(state)
      ret.append((feats['dist'], feats['angle']))

      if extractor.label != 'segs':
        ret.append((feats['dist2'], feats['angle2']))
      else:
        ret.append((feats['curDist'], feats['curAngle']))
    return (ret, action)

  return getDistAngelList

def getHumanDiscreteMapper(mdp, category = None):
  """
  Return ((targDist, targAngle)_1^2, (obstDist, obstAngle)_1^2, (segDist, segAngle)) and action
  """
  if category == None:
    # assume need all the classes 
    extractors = [HumanViewExtractor(mdp, label) for label in ['targs', 'obsts', 'segs']]
  else:
    extractors = [HumanViewExtractor(mdp, category)]

  def getDistAngelList(state, action):
    states = []
    for extractor in extractors:
      feats = extractor.getStateFeatures(state)
      
      state, action = discreteQTableCompressor((feats['dist'], feats['angle']), action)
      states.append(state)
      if not extractor.label == 'segs':
        # add second closest objects
        if feats['dist2'] != None and feats['angle2'] != None:
          state, action = discreteQTableCompressor((feats['dist2'], feats['angle2']), action)
          states.append(state)
        else:
          states.append((None, None))

    return (states, action)
  
  if category != None:
    uncoupleState = lambda (s, a): (s[0], a) 
    ret = lambda s, a: uncoupleState(getDistAngelList(s, a))
  else:
    ret = getDistAngelList

  return ret

def gridGetNext(mdp, state, action):
  # simulate the next state
  x, y = state
  dx, dy = Actions.directionToVector(action)
  next_x, next_y = int(x + dx), int(y + dy)
  if next_x < 0 or next_x >= mdp.grid.width:
    next_x = x
  if next_y < 0 or next_y >= mdp.grid.height:
    next_y = y

  return [next_x, next_y]

def getGridMapper(mdp):
  moduleClasses = map(lambda _: _[0], mdp.spec)

  def getDists(state, action):
    """
    Compute a Q value responding to an object, considering the distance to it.
    This is used by obstacle avoidance, and target obtaining.

    Args:
      r: the reward of the module class to be found
      idx: the id of the class
    """
    states = []
    x, y = state
    # find the distance to the nearest object
    for moduleClass in moduleClasses:
      dists = []
      for xt in range(mdp.grid.width):
        for yt in range(mdp.grid.height):
          cell = mdp.grid[xt][yt] 
          if cell == moduleClass:
            dist = np.sqrt((xt - x) ** 2 + (yt - y) ** 2)
            dists.append(dist)
      states.append(dists)
    return (states, action)

  return getDists

def discreteQTableCompressor(state, action):
  dist, angle = state
  
  if dist == None or angle == None:
    return state

  newAction = action
  if action == 'G':
    # force table to be symmetric
    angle = abs(angle)
  elif action == 'L':
    # use R table
    angle = -angle
    newAction = 'R'
  elif action == 'SL':
    # use SR table
    angle = -angle
    newAction = 'SR'
  
  newState = mapStateToBin((dist, angle))
  return (newState, newAction)

"""
Some util functions for feature extraction.
"""
def getClosestObj(loc, l):
  """
  Args:
    loc: location of the agent
    l: list of objects
  """
  minDist = np.inf
  minObj = loc

  for obj in l:
    dist = numpy.linalg.norm(np.subtract(loc, obj))
    if dist < minDist:
      minDist = dist
      minObj = obj

  return [minObj, minDist]

def getProjectionToSegment(loc, segs):
  """
  Compute the projection from loc to the segment with vertices of seg0 and seg1
  """
  if len(segs) == 0:
    return [loc, np.inf]
  elif len(segs) == 1:
    seg = segs[0]
    return [seg, numpy.linalg.norm(np.subtract(loc, seg))]
  else:
    # FIXME better way to compute projection?
    from shapely.geometry import LineString, Point
    segVec = np.subtract(segs[1], segs[0])
    line = LineString([np.add(segs[0], - 100 * segVec), np.add(segs[0], 100 * segVec)])
    p = Point(loc)
    interceptPoint = line.interpolate(line.project(p))
    intercept = (interceptPoint.x, interceptPoint.y)
    return intercept

def getProjectionToSegmentLocalView(s0, s1):
  """
  If we only have distance, angle to the segments, use this function.
  This will call getProjectionToSegment.
  """
  loc = (0, 0)
  segs = [(dist * np.cos(orient), dist * np.sin(orient)) for (dist, orient) in [s0, s1]]
  obj = getProjectionToSegment(loc, segs)
  return getDistAngle(loc, obj, 0)

def getDistAngle(f, t, orient):
  if t == None:
    return (None, None)
  else:
    vector = np.subtract(t, f)
    dist = numpy.linalg.norm(vector)
    objOrient = np.angle(vector[0] + vector[1] * 1j)
    return [dist, adjustAngle(objOrient - orient)]

def getSortedObjs(loc, l):
  """
  Sort l out-of-place wrt the distance to loc
  """
  newl = list(l)
  newl.sort(key = lambda obj : numpy.linalg.norm(np.subtract(loc, obj)))
  return newl

 
def adjustAngle(angle):
  while angle < - np.pi:
    angle += 2 * np.pi
  while angle > np.pi:
    angle -= 2 * np.pi
  return angle

distances = [.1, .2, .3, .5, .75, 1, 1.5, 2, 2.5, 10]
angles = [-90, -60, -30, -10, -5, -2, 0, 2, 5, 10, 30, 60, 90, 181] # human readable

# original setting
"""
distances = map(lambda _: _ * humanWorld.HumanWorld.step, [.5, 1, 1.5, 2, 2.5, 3, 4, 10])
angles = [-135, -90, -60, -30, -20, -10, 0, 10, 20, 30, 60, 90, 135, 180]
"""

anglesArc = map(lambda x: 1.0 * x / 180 * np.pi, angles)

def mapStateToBin((dist, angle)):
  if dist == None or angle == None:
    return (dist, angle)

  distBin = len(distances)
  for idx in xrange(len(distances)):
    if dist < distances[idx]:
      distBin = idx
      break
    
  angleBin = len(distances)
  for idx in xrange(len(anglesArc)):
    if angle < anglesArc[idx]:
      angleBin = idx
      break

  if distBin == len(distances):
    warnings.warn("distance too long: " + str(dist))
  if angleBin == len(angles):
    raise Exception('observing unexpected angle of ' + str(angle))

  return (distBin, angleBin)

def binsGaussianKernel(key):
  """
  Discrete approximation of gausian kernel.
  
  key -> {key : weight}
  """
  (distBin, angleBin), action = key

  retSet = util.Counter()
  retSet[key] = 4
  # don't worry about edges
  if distBin > 0 and distBin < len(distances) - 1 and angleBin > 0 and angleBin < len(angles) - 1:
    retSet[(distBin - 1, angleBin), action] = 2
    retSet[(distBin + 1, angleBin), action] = 2
    retSet[(distBin, angleBin - 1), action] = 2
    retSet[(distBin, angleBin + 1), action] = 2

    retSet[(distBin - 1, angleBin - 1), action] = 1
    retSet[(distBin - 1, angleBin + 1), action] = 1
    retSet[(distBin + 1, angleBin - 1), action] = 1
    retSet[(distBin + 1, angleBin + 1), action] = 1
  
  normalizer = sum(retSet.values()) 
  normedRetSet = {key: 1.0 * value / normalizer for key, value in retSet.items()}
  return normedRetSet
