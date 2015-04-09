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
    ret = []
    for extractor in extractors:
      feats = extractor.getStateFeatures(state)
      
      newAction = action
      if action == 'G':
        angleMap = lambda angle: abs(angle)
      elif action == 'L':
        angleMap = lambda angle: -angle
        newAction = 'R'
      elif action == 'R':
        angleMap = lambda angle: angle
      else:
        raise Exception('Unknown action ' + action)
        
      state = (feats['dist'], angleMap(feats['angle']))
      ret.append(mapStateToBin(state))
      if not extractor.label == 'segs':
        # add second closest objects
        state = (feats['dist2'], feats['angle2'])
        ret.append(mapStateToBin(state))

    return (ret, newAction)

  return getDistAngelList

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

def mapStateToBin((dist, angle)):
  if dist == None or angle == None:
    return (dist, angle)
  # FIXME OVERFIT
  if dist < 0.1:
    distBin = 1
  elif dist < 0.2:
    distBin = 2
  elif dist < 0.3:
    distBin = 3
  elif dist < 0.5:
    distBin = 4
  elif dist < 0.75:
    distBin = 5
  elif dist < 1:
    distBin = 6
  elif dist < 1.5:
    distBin = 7
  elif dist < 2:
    distBin = 8
  elif dist < 2.5:
    distBin = 9
  else:
    distBin = 10

  if abs(angle) < 2.0 / 180 * np.pi:
    angleBin = 0
  elif abs(angle) < 5.0 / 180 * np.pi:
    angleBin = int(1 * np.sign(angle))
  elif abs(angle) < 10.0 / 180 * np.pi:
    angleBin = int(2 * np.sign(angle))
  elif abs(angle) < 30.0 / 180 * np.pi:
    angleBin = int(3 * np.sign(angle))
  elif abs(angle) < 60.0 / 180 * np.pi:
    angleBin = int(4 * np.sign(angle))
  elif abs(angle) < 90.0 / 180 * np.pi:
    angleBin = int(5 * np.sign(angle))
  else:
    angleBin = int(6 * np.sign(angle))

  return (distBin, angleBin)
