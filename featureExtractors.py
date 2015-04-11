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

def discreteQTableCompressor(state, action):
  dist, angle = state

  newAction = action
  """
  if action == 'G':
    # force table to be symmetric
    angle = abs(angle)
  elif action == 'L':
    # use R table
    angle = -angle
    newAction = 'R'
  elif action == 'R':
    angle = angle
  """
  
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

distances = [.1, .2, .3, .5, .75, 1, 1.5, 2, 2.5, 5]
angles = [-90, -60, -30, -10, -5, -2, 0, 2, 5, 10, 30, 60, 90, 181]

def mapStateToBin((dist, angle)):
  if dist == None or angle == None:
    return (dist, angle)

  if dist > distances[-1]:
    raise Exception('observing unexpected distance of ' + str(dist))
  if angle > angles[-1]:
    raise Exception('observing unexpected angle of ' + str(angle))

  for idx in xrange(len(distances)):
    if dist < distances[idx]:
      distBin = idx
      break
    
  anglesArc = map(lambda x: 1.0 * x / 180 * np.pi, angles)
  for idx in xrange(len(anglesArc)):
    if angle < anglesArc[idx]:
      angleBin = idx
      break

  return (distBin, angleBin)
