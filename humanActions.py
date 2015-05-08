import numpy as np
import collections
import config

class HumanActions:
  def __init__(self, actions, dists, angles):
    """
    Pass in actions and the expected angles
    """
    assert len(actions) == len(angles)
    self.actionDists = collections.OrderedDict(zip(actions, dists))
    self.actionAngles = collections.OrderedDict(zip(actions, angles))
    
  def getActions(self):
    return self.actionAngles.keys()

  def angleToAction(self, angle):
    assert -np.pi < angle and angle < np.pi
    
    actions = self.actionAngles.keys()
    angles = self.actionAngles.values()
    for i in xrange(len(self.actionAngles) - 1):
      if angle < (angles[i] + angles[i + 1]) / 2:
        return actions[i]
    return actions[-1]
  
  def getExpectedDistAngle(self, action):
    assert action in self.actionAngles.keys()

    return [self.actionDists[action], self.actionAngles[action]]
  
def getNarrowedHumanActions(step):
  """
  Assume human basically only moves forward.
  """
  # angles for turning actions
  turnAngle = 45.0 / 180 * np.pi
  slightTurnAngle = 15.0 / 180 * np.pi
  turnDist = step * 1 / 3
  slightTurnDist = step * 2 / 3
  walkDist = step * 1

  if config.SLIGHT_TURNS:
    return HumanActions(('L', 'SL', 'G', 'SR', 'R'),\
                        (turnDist, slightTurnDist, walkDist, slightTurnDist, turnDist),\
                        (-turnAngle, -slightTurnAngle, 0, slightTurnAngle, turnAngle))
  else:
    return HumanActions(('L', 'G', 'R'),\
                        (turnDist, walkDist, turnDist),\
                        (-turnAngle, 0, turnAngle))

def getBroadHumanActions(step):
  """
  Assume human can move all forward directions
  """
  actions = range(-3, 4)
  angles = [1.0 * actionId * 30 / 180 * np.pi for actionId in actions]
  dists = [step / 2] * len(actions)
  
  return HumanActions(actions, dists, angles)