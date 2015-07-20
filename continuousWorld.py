import random
from game import Actions
import mdp
import mdpEnvironment
import util
import featureExtractors

import numpy as np
import numpy.linalg
import config

class ContinuousWorld(mdp.MarkovDecisionProcess):
  """
  A MDP that captures continuous state space, while the agent moves in discrete steps.

  State: (location, orientation) of the agent
  Action: 8 directional movement, with a fixed step size.
  Transition: trivial.
  Reward: upon reaching a target / obstacle, obtain the corresponding reward.
          upon reaching a segment, if this segment is newer than the one visited before,
          then obtain the reward.
  """
  def __init__(self, init):
    """
    Args:
      init: a dict, that has attributes to be appended to self
            objs, boundary, radius.
    """
    # this adds necessary attributes of this MDP from a domain initializer
    # domains are coded in continuousWorldDomains
    self.__dict__.update(init)

    # reward values that getReward will use
    self.rewards = {'targs': 1, 'obsts': -1, 'segs': 1, 'elevators': 0, 'entrance': 0}
    self.noise = 0.0 # DUMMY - need assumption on what it means to be noisy

    # stats set
    self.touchedObstacleSet = []
    self.collectedTargetSet = []

    # stateWindow is the window that states are sampled from,
    # smaller or equal to the original world window
    if not 'stateWindow' in self.__dict__.keys():
      self.stateWindow = [self.xBoundary[0], self.xBoundary[1], self.yBoundary[0], self.yBoundary[1]]
    if not 'livingReward' in self.__dict__.keys():
      self.livingReward = 0

  def getReachedObjects(self, l):
    """
    Determine whether a state is close to any object, within radius.
    The radius for obstacles should be larger than that of targets.
    A waypoint is 'reached' when the agent gets closer to the next waypoint.

    Args:
      l: the loc to be checked.
    Return:
      [(String, int)]: A list of the type of object that it's in the radius of, and its id.
                       Return [] if nothing matches.
    """
    ret = []

    # get a target?
    tLocs = self.objs['targs']
    for targIdx in xrange(len(tLocs)):
      dist = numpy.linalg.norm(np.subtract(l, tLocs[targIdx]))
      if dist < self.radius:
        ret.append(('targs', targIdx))

    # run into an obstacle?
    oLocs = self.objs['obsts']
    for obstIdx in xrange(len(tLocs)):
      dist = numpy.linalg.norm(np.subtract(l, oLocs[obstIdx]))
      # using larger obstacle when training?
      scale = 2 if config.TRAINING else 1
      if dist < self.radius * scale:
        ret.append(('obsts', obstIdx))

    sLocs = self.objs['segs']
    segIdx = 0
    # close to the next segment then remove the current one
    while (segIdx < len(sLocs)):
      if segIdx < len(sLocs) - 1:
        # when get closer to the next one
        distSeg1 = numpy.linalg.norm(np.subtract(l, sLocs[segIdx]))
        distSeg2 = numpy.linalg.norm(np.subtract(l, sLocs[segIdx + 1]))
        if distSeg1 > distSeg2 :
          ret.append(('segs', segIdx))
        else:
          break
      else:
        # if only one left, just approach it
        distSeg = numpy.linalg.norm(np.subtract(l, sLocs[segIdx]))
        if distSeg < self.radius * 2: # larger buffer
          ret.append(('segs', segIdx))
          break
      
      segIdx += 1
    """
    # see it as waypoint collection
    while (segIdx < len(sLocs)):
      distSeg = numpy.linalg.norm(np.subtract(l, sLocs[segIdx]))
      if distSeg < self.radius * 3.5: # larger buffer
        ret.append(('segs', segIdx))
        segIdx += 1
      else:
        break
    """

    return ret

  def setLivingReward(self, reward):
    """
    The (negative) reward for exiting "normal" states.
    
    Note that in the R+N text, this reward is on entering
    a state and therefore is not clearly part of the state's
    future rewards.
    """
    self.livingReward = reward
        
  def setNoise(self, noise):
    """
    The probability of moving in an unintended direction.
    """
    self.noise = noise
                                    
  def getPossibleActions(self, state):
    """
    Returns list of valid actions for 'state'.

    This discretizes possible aciton set.
    """
    return ('north','west','south','east', 'ne', 'se', 'nw', 'sw')
    
  def getStates(self):
    """
    Return list of random states. This is usually for sanity check.
    """
    states = []
    windowX = self.stateWindow[0:2]
    windowY = self.stateWindow[2:4]

    stepX = (windowX[1] - windowX[0]) / 25
    stepY = (windowY[1] - windowY[0]) / 25
    
    x = windowX[0]
    while x < windowX[1]:
      y = windowY[0]
      while y < windowY[1]:
        states.append(((x, y), 0))
        y += stepY
      x += stepX
      
    return states
        
  def getReward(self, state, action, nextState):
    """
    Get reward for state, action, nextState transition.
    
    Note that the reward depends only on the state being
    departed (as in the R+N book examples, which more or
    less use this convention).
    """
    loc, orient = state
    nextLoc, nextOrient = nextState
    
    reward = 0

    # get the list of contacted objects
    objInfoList = self.getReachedObjects(nextLoc)

    for nextStateType, nextObjId in objInfoList:
      if nextStateType != 'segs':
        # add rewards for target and obstacles
        reward += self.rewards[nextStateType]

        # keep a set of reached objects
        objLoc = self.objs[nextStateType][nextObjId]
        if nextStateType == 'obsts' and not objLoc in self.touchedObstacleSet:
          self.touchedObstacleSet.append(objLoc)
        elif nextStateType == 'targs':
          self.collectedTargetSet.append(objLoc)

    # give rewards for waypoint segments, except training targets or obstacles
    if not hasattr(self, 'category') or self.category == 'segs': 
      nextSeg = self.objs['segs'][1] if len(self.objs['segs']) > 1 else self.objs['segs'][0]

      dist, angle = featureExtractors.getDistAngle(loc, nextSeg, orient)
      nextStepDist, nextStepAngle = featureExtractors.getDistAngle(nextLoc, nextSeg, orient)
      [pathIntercept, pathDist] = featureExtractors.getProjectionToSegment(nextLoc, self.objs['segs'])

      # proposed by Matt
      # discounted reward for shrinking distance and not far from the path
      # don't punish it. return at least 0.
      if nextStepDist < dist:
        reward += max(self.rewards['segs'] * (1 - 2 * pathDist ** 2), 0)

    return reward or self.livingReward
      
  def clearObj(self, objType, objId):
    """
    Clear an object from self.objs, usually because the agent has got it.
    """
    del self.objs[objType][objId]

  def getStartState(self):
    """
    Start at the starting location, with no segment previously visited.
    """
    loc = self.objs['entrance']

    # face towards the center of the domain
    if loc[0] < 0: angle = 45.0 / 180 * np.pi
    else: angle = - 135.0 / 180 * np.pi
    return (loc, angle)
    
  def isFinal(self, state):
    """
    Check whether we should terminate at this state.
    """
    loc, orient = state

    if config.TRAINING:
      return len(self.objs['segs']) == 0 or len(self.objs['targs']) == 0
    else:
      return len(self.objs['segs']) == 0

  def getTransitionStatesAndProbs(self, state, action):
    """
    Basically following physics laws, but considering:
    - stay within self.xBoundary and self.yBoundary (may bump into the boundary)
    - change seg in the state representation upon reaching a new segment
    """
    loc, orient = state

    # move to new loc and check whether it's allowed
    newLoc = np.add(loc, np.multiply(self.step, Actions._directions[action]))
    if not self.isAllowed(newLoc):
      newLoc = loc

    newOrient = orient # doesn't change orient in this mdp
    
    successors = [((newLoc, newOrient), 1)]

    return successors                                
  
  def __aggregate(self, statesAndProbs):
    """
    Make sure stateAndProbs.keys() is a set (without duplicates)
    """
    counter = util.Counter()
    for state, prob in statesAndProbs:
      counter[state] += prob
    newStatesAndProbs = []
    for state, prob in counter.items():
      newStatesAndProbs.append((state, prob))
    return newStatesAndProbs
        
  def isAllowed(self, loc):
    """
    Check whether this state is valid
    """
    x, y = loc
    if x < self.xBoundary[0] or x >= self.xBoundary[1]: return False
    if y < self.yBoundary[0] or y >= self.yBoundary[1]: return False
    return True

class ContinuousEnvironment(mdpEnvironment.MDPEnvironment):
  def step(self, state, action, nextState, reward):
    # remove objects if necessary
    # clear this object upon getting it
    loc, orient = state
    nextLoc, nextOrient = nextState

    objInfoLists = self.mdp.getReachedObjects(nextLoc)
    # to remove object by id, make sure remove the ones with larger id first
    # so the list won't screw up.
    if len(objInfoLists) > 0: objInfoLists.reverse()

    for nextStateType, nextObjId in objInfoLists:
      if nextStateType == 'targs' or nextStateType == 'segs':
        self.mdp.clearObj(nextStateType, nextObjId)

