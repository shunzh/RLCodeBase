import featureExtractors
import continuousWorld

import numpy as np
import config

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
  # static attributes
  step = 0.3

  # angles for turning actions
  turnAngle = 30.0 / 180 * np.pi
  turnDist = step / 3
  walkDist = step * 1
  # angles smaller than turnAngleThreshold classified as G
  turnAngleThreshold = turnAngle / 2

  actions = ('L', 'R', 'G')

  if config.SLIGHT_TURNS:
    turnAngle = 45.0 / 180 * np.pi
    slightTurnAngle = 15.0 / 180 * np.pi
    slightTurnDist = step * 2 / 3

    slightTurnAngleThreshold = slightTurnAngle / 2
    turnAngleThreshold = (slightTurnAngle + turnAngle) / 2

    actions += ('SL', 'SR')
    
  def getPossibleActions(self, state):
    return HumanWorld.actions

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

    if action == 'L':
      newOrient = orient - self.turnAngle
      d = self.turnDist
    elif action == 'R':
      newOrient = orient + self.turnAngle
      d = self.turnDist
    # slight turns may not be enabled
    elif action == 'SL':
      newOrient = orient - self.slightTurnAngle
      d = self.turnDist
    elif action == 'SR':
      newOrient = orient + self.slightTurnAngle
      d = self.turnDist
    elif action == 'G':
      newOrient = orient
      d = self.walkDist
    else:
      raise Exception("Unknown action.")
  
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
  def angleToAction(moveAngle):
    """
    Move angle is parsed from mat files.
    Classify angles into actions according to our thresholds
    """
    if moveAngle < -HumanWorld.turnAngleThreshold:
      action = 'L'
    elif moveAngle < HumanWorld.turnAngleThreshold:
      if config.SLIGHT_TURNS:
        if moveAngle < -HumanWorld.slightTurnAngleThreshold:
          action = 'SL'
        elif moveAngle < HumanWorld.slightTurnAngleThreshold:
          action = 'G'
        else:
          action = 'SR'
      else:
        action = 'G'
    else:
      action = 'R'
    
    return action
    
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
    turnAngle = HumanWorld.turnAngle
    if config.SLIGHT_TURNS: slightTurnAngle = HumanWorld.slightTurnAngle
    turnDist = HumanWorld.turnDist
    walkDist = HumanWorld.walkDist

    dist, orient = s

    objX = dist * np.cos(orient) 
    objY = dist * np.sin(orient) 
    
    if a == 'G':
      orient = 0
      aX = walkDist; aY = 0
    else:
      if a == 'L':
        orient = -turnAngle
      elif a == 'R':
        orient = turnAngle
      elif a == 'SL':
        orient = -slightTurnAngle
      elif a == 'SR':
        orient = slightTurnAngle

      aX = turnDist * np.cos(orient) 
      aY = turnDist * np.sin(orient) 
    
    # the new state is from (aX, aY) to (objX, objY
    newDist, newOrient = featureExtractors.getDistAngle((aX, aY), (objX, objY), orient)

    return (newDist, newOrient)

# The environment is same as the continuousWorld
HumanEnvironment = continuousWorld.ContinuousEnvironment

