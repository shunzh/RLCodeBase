import random
import sys
import mdp
import environment
import util
import optparse
import featureExtractors
import continuousWorld

import numpy as np
import numpy.linalg
import warnings

from graphics import *

class humanWorld(continuousWorld.ContinuousWorld):
  """
  An MDP that agrees with Matt's human data.

  It is almost the same as ContinuousWorld, but this is in the agent's view.
  The agent stays in (0, 0) forever, while the distance / angle to the object changes.

  State: (location, ref to obj)
  Action: L, R, G
  Transition: same as continuousWorld but 
  Reward: same as continuousWorld.
  """
  def getPossibleActions(self, state):
    """
    L: Turn left 30 degrees and walk ahead 0.05m.
    R: Turn right 30 degrees and walk ahead 0.05m.
    G: Go ahead 0.2m.
    """
    return ('L', 'R', 'G')
