import featureExtractors
import continuousWorld

import numpy as np
import config
import humanActions

class HumanWorld(continuousWorld.ContinuousWorld):
  """
  An MDP that agrees with Matt's human data.

  It is almost the same as ContinuousWorld, but this is in the agent's view.
  The agent uses the distance / angle to the object as state.
  
  The coordinates: vectors (visually) above the x-axis have negative angles.
  This is to be consistent with the parsed mat files.

      -pi/2
  -pi   |
  ------------- 0
   pi   |
       pi/2

  State: (distance, orient) for targ, obst, seg, respectively
  Action: L, R, SL, SR, G
  Transition: same as continuousWorld but 
  Reward: same as continuousWorld.
  """
  step = 0.3
  actions = humanActions.getNarrowedHumanActions(step)

  def getPossibleActions(self, state = None):
    return HumanWorld.actions.getActions()

  def __init__(self, init):
    continuousWorld.ContinuousWorld.__init__(self, init)

    self.atBorder = False
  
  def isFinal(self, state):
    """
    The transition at the border may be unexpected, which may cause the q values updated incorrectly.
    For example, G action towards the border makes the agent stay within the domain, but q value of 
    such state is still updated.
    So terminate the simulation if running into the border.
    """
    return continuousWorld.ContinuousWorld.isFinal(self, state) or self.atBorder

  def getTransitionStatesAndProbs(self, state, action):
    """
    Responde to actions.

    Use: self.[turnAngle, turnDist, walkDist]
    """
    loc, orient = state

    d, turnAngle = HumanWorld.actions.getExpectedDistAngle(action)
    newOrient = orient + turnAngle
  
    newOrient = featureExtractors.adjustAngle(newOrient)

    dv = (d * np.cos(newOrient), d * np.sin(newOrient))
    newLoc = np.add(loc, dv)
    if not self.isAllowed(newLoc):
      self.atBorder = True
      newLoc = loc
    else:
      self.atBorder = False

    newLoc = tuple(newLoc) # make sure type is consistent

    newState = (newLoc, newOrient)

    return [(newState, 1)]
  
  @staticmethod
  def transitionSimulate(s, a):
    """
    In some cases, we need to get the next state given the current state and action.
    Because the current state is represented by distance and angle to an object,
    it's a bit tricky to get the distance and angle to the object after taking an action.

    This is done by creating an ad-hoc coordinate space and do a one step simulation.

    Args:
      s: (dist, orient)
      a: action
    Return:
      (newDist, newOrient) after taking a in state s.
    """
    # use human world info for simulation
    dist, orient = s
    
    if dist == None or orient == None:
      return (None, None)

    objX = dist * np.cos(orient) 
    objY = dist * np.sin(orient) 
    
    dist, orient = HumanWorld.actions.getExpectedDistAngle(a)
    aX = dist * np.cos(orient) 
    aY = dist * np.sin(orient) 
    
    # the new state is from (aX, aY) to (objX, objY
    newDist, newOrient = featureExtractors.getDistAngle((aX, aY), (objX, objY), orient)

    return (newDist, newOrient)

# The environment is same as the continuousWorld
HumanEnvironment = continuousWorld.ContinuousEnvironment