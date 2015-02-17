# featureExtractors.py
# --------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"Feature extractors for Pacman game states"

from game import Directions, Actions
import util
import math

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


def getSortedObjs(loc, l):
  """
  Sort l out-of-place wrt the distance to loc
  """
  newl = list(l)
  newl.sort(key = lambda obj : numpy.linalg.norm(np.subtract(loc, obj)))
  return newl


class ContinousRadiusLogExtractor(FeatureExtractor):
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
  From human's view
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

    def getOrient(f, t):
      """
      Compute the orient from f to t, both are points
      """
      vector = np.subtract(t, f)
      objOrient = np.angle(vector[0] + vector[1] * 1j)
      return adjustAngle(objOrient - orient)

    if self.label == 'segs':
      # get features for waypoints
      if len(self.mdp.objs['segs']) > 1:
        minObj = self.mdp.objs['segs'][1] # look at the NEXT waypoint
        minDist = numpy.linalg.norm(np.subtract(loc, minObj))
      elif len(self.mdp.objs['segs']) == 1:
        minObj = self.mdp.objs['segs'][0] # this is the last segment
        minDist = numpy.linalg.norm(np.subtract(loc, minObj))
      else:
        minObj = loc; minDist = np.inf

      feats['dist'] = minDist
      feats['angle'] = getOrient(loc, minObj)
    else:
      # get features for targets / objects
      # get both closest and the second closest -- may not be both used though
      l = getSortedObjs(loc, self.mdp.objs[self.label])
      if len(l) > 0:
        minObj = l[0]
        minDist = numpy.linalg.norm(np.subtract(loc, minObj))
      else:
        minObj = loc; minDist = np.inf

      if len(l) > 1:
        # if there are more than two objects
        secMinObj = l[1]
        secMinDist = numpy.linalg.norm(np.subtract(loc, secMinObj))
      else:
        secMinObj = loc; secMinDist = np.inf

      feats['dist'] = minDist
      feats['angle'] = getOrient(loc, minObj)
      feats['dist2'] = secMinDist
      feats['angle2'] = getOrient(loc, secMinObj)

    feats['bias'] = 1

    return feats

def getHumanContinuousState(mdp):
  """
  Return ((targDist, targAngle), (obstDist, obstAngle), (segDist, segAngle))
  """
  extractors = [HumanViewExtractor(mdp, label) for label in ['targs', 'obsts', 'segs']]

  def getDistAngelList(state):
    ret = []
    for extractor in extractors:
      feats = extractor.getStateFeatures(state)
      ret.append((feats['dist'], feats['angle']))
      if not extractor.label == 'segs':
        ret.append((feats['dist2'], feats['angle2']))
    return ret

  return getDistAngelList
  
def adjustAngle(angle):
  while angle < - np.pi:
    angle += 2 * np.pi
  while angle > np.pi:
    angle -= 2 * np.pi
  return angle

def mapStateToBin((dist, angle), step):
  # FIXME OVERFIT
  if dist < step * 1:
    distBin = 1
  elif dist < step * 2:
    distBin = 2
  elif dist < step * 3:
    distBin = 3
  elif dist < step * 4:
    distBin = 4
  elif dist < step * 5:
    distBin = 5
  elif dist < step * 8:
    distBin = 6
  elif dist < step * 10:
    distBin = 7
  elif dist < step * 15:
    distBin = 8
  elif dist < step * 20:
    distBin = 9
  else:
    distBin = 10

  if abs(angle) < 15.0 / 180 * np.pi:
    angleBin = 0
  elif abs(angle) < 45.0 / 180 * np.pi:
    angleBin = int(1 * np.sign(angle))
  elif abs(angle) < 90.0 / 180 * np.pi:
    angleBin = int(2 * np.sign(angle))
  elif abs(angle) < 135.0 / 180 * np.pi:
    angleBin = int(3 * np.sign(angle))
  else:
    angleBin = int(4 * np.sign(angle))

  return (distBin, angleBin)

def getHumanViewBins(mdp, label):
  """
  Get bins extracted from continuous features.
  """
  extractor = HumanViewExtractor(mdp, label)

  def getBins(state):
    feats = extractor.getStateFeatures(state)
    if feats == []:
      return ()

    return mapStateToBin((feats['dist'], feats['angle']), mdp.step)

  return getBins

def getHumanDiscreteState(mdp):
  """
  Return ((targDist, targAngle)_1^2, (obstDist, obstAngle)_1^2, (segDist, segAngle))
  """
  extractors = [HumanViewExtractor(mdp, label) for label in ['targs', 'obsts', 'segs']]

  def getDistAngelList(state):
    ret = []
    for extractor in extractors:
      feats = extractor.getStateFeatures(state)
      ret.append(mapStateToBin((feats['dist'], feats['angle']), mdp.step))
      if not extractor.label == 'segs':
        # add second closest objects
        ret.append(mapStateToBin((feats['dist2'], feats['angle2']), mdp.step))

    return ret

  return getDistAngelList
 
class ObstacleExtractor(FeatureExtractor):
  """
  This should use radius extractor.
  """
  def getFeatures(self, state, action):
    feats = util.Counter()
    feats['bias'] = 1

    x, y = state
    dx, dy = Actions.directionToVector(action)
    # distance to obstacle, on x and y
    disx = x + dx - 2
    disy = y + dy - 2

    feats['dis'] = math.sqrt(disx*disx + disy*disy)
    
    return feats
	
class SidewalkExtractor(FeatureExtractor):
  def getFeatures(self, state, action):
    feats = util.Counter()

    x, y = state
    dx, dy = Actions.directionToVector(action)
    feats['x'] = x + dx
    
    return feats

def closestFood(pos, food, walls):
  """
  closestFood -- this is similar to the function that we have
  worked on in the search project; here its all in one place
  """
  fringe = [(pos[0], pos[1], 0)]
  expanded = set()
  while fringe:
    pos_x, pos_y, dist = fringe.pop(0)
    if (pos_x, pos_y) in expanded:
      continue
    expanded.add((pos_x, pos_y))
    # if we find a food at this location then exit
    if food[pos_x][pos_y]:
      return dist
    # otherwise spread out from the location to its neighbours
    nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
    for nbr_x, nbr_y in nbrs:
      fringe.append((nbr_x, nbr_y, dist+1))
  # no food found
  return None

class SimpleExtractor(FeatureExtractor):
  """
  Returns simple features for a basic reflex Pacman:
  - whether food will be eaten
  - how far away the next food is
  - whether a ghost collision is imminent
  - whether a ghost is one step away
  """
  
  def getFeatures(self, state, action):
    # extract the grid of food and wall locations and get the ghost locations
    food = state.getFood()
    walls = state.getWalls()
    ghosts = state.getGhostPositions()

    features = util.Counter()
    
    features["bias"] = 1.0
    
    # compute the location of pacman after he takes the action
    x, y = state.getPacmanPosition()
    dx, dy = Actions.directionToVector(action)
    next_x, next_y = int(x + dx), int(y + dy)
    
    # count the number of ghosts 1-step away
    features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

    # if there is no danger of ghosts then add the food feature
    if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
      features["eats-food"] = 1.0
    
    dist = closestFood((next_x, next_y), food, walls)
    if dist is not None:
      # make the distance a number less than one otherwise the update
      # will diverge wildly
      features["closest-food"] = float(dist) / (walls.width * walls.height) 
    features.divideAll(10.0)
    return features
