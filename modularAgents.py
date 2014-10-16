from qlearningAgents import ApproximateQAgent
from game import Actions
import util
import featureExtractors

import math
import numpy as np
import numpy.linalg

class ModularAgent(ApproximateQAgent):
  """
  State: location of the agent.
  Action: weights on the sub-MDPs.
  Transition: transition of the agent.
  Reward: reward from the environment.

  Assume:
  Weights are independent from the location of the agent.
  """
  def __init__(self, **args):
    ApproximateQAgent.__init__(self, **args)

    # assume the weights are not dynamically learned, intialize them here.
    self.weights = [1, 0, 0]
    self.learningWeights = False
 
  def getQValue(self, state, action):
    """
    Get Q value by consulting each module
    """
    # sum over q values from each sub mdp
    return sum([self.qFuncs[i](state, action) * self.weights[i] for i in xrange(len(self.qFuncs))])

  def getSubQValues(self, state, action):
    return [self.qFuncs[i](state, action) for i in xrange(len(self.qFuncs))]

  def getSoftmaxQValue(self, state, action):
    actions = self.getLegalActions(state)
    j = actions.index(action)

    # q value matrix
    # vMat(i, j) is for i-th module and j-th action
    vMat = []

    # iterate through all the modules
    for qFunc in self.qFuncs:
      # list of exp^q
      exps = [math.exp(qFunc(state, action)) for action in actions]

      # Normalize
      sumExps = sum(exps)
      vMat.append([exp / sumExps for exp in exps])

    # sum over j-th column (j-th action)
    return sum([vMat[i][j] for i in range(len(self.qFuncs))])

  def setQFuncs(self, qFuncs):
    """
    Set QFuncs from the environment. getQValue will use this.
    """
    self.qFuncs = qFuncs

  def setWeights(self, weights):
    """
    If this is set, then the agent simply follows the weights.
    """
    self.learningWeights = False
    self.weights = weights

  def getWeights(self):
    """
    Get weights.
    """
    return self.weights
  
  def update(self, state, action, nextState, reward):
    if not self.learningWeights:
      pass
    else:
      # TODO
      raise Exception("calling unimplemented method: update for learning weight.")

  def getSignificance(self, state):
    """
    How significance an agent's correct decision at this state should affect the overall performance.

    Using the std of the Q values of this state.

    DUMMY
    """
    actions = self.getLegalActions(state)

    values = [self.getQValue(state, action) for action in actions]
    return np.std(values)


def getObsAvoidFuncs(mdp):
  """
  Return Q functions for modular mdp for obstacle avoidance behavior

  the environment is passed by mdp
  """
  obstacle = {'bias': -0.20931133310480204, 'dis': 0.06742681562641269}
  target = {'bias': 0.20931133310480204, 'dis': -0.06742681562641269}
  sidewalk = {'x': 0.06250000371801567}

  def getNext(state, action):
    x, y = state
    dx, dy = Actions.directionToVector(action)
    next_x, next_y = int(x + dx), int(y + dy)
    if next_x < 0 or next_x >= mdp.grid.width:
      next_x = x
    if next_y < 0 or next_y >= mdp.grid.height:
      next_y = y

    return [next_x, next_y]

  def qWalk(state, action):
    """
    QValue of forward walking
    """
    next_x, next_y = getNext(state, action)
    return sidewalk['x'] * next_x

  def radiusBias(state, action, cond, w):
    """
    Compute a Q value responding to an object, considering the distance to it.
    This is used by obstacle avoidance, and target obtaining.

    Args:
      state, action
      cond: the lambda expr that given state is the object we want
      w: weight vector
    """
    x, y = state
    next_x, next_y = getNext(state, action)

    # find the distance to the nearest object
    minDist = mdp.grid.width * mdp.grid.height
    for xt in range(mdp.grid.width):
      for yt in range(mdp.grid.height):
        cell = mdp.grid[xt][yt] 
        if cond(cell):
          # it's an obstacle!
          dist = math.sqrt((xt - next_x) ** 2 + (yt - next_y) ** 2)
          if (dist < minDist): minDist = dist
    return minDist * w['dis'] + 1 * w['bias']

  def qObstacle(state, action):
    cond = lambda s : (type(s) == int or type(s) == float) and s == -1
    return radiusBias(state, action, cond, obstacle)

  def qTarget(state, action):
    cond = lambda s : (type(s) == int or type(s) == float) and s == +1
    return radiusBias(state, action, cond, target)

  return [qWalk, qObstacle, qTarget]


def getContinuousWorldFuncs(mdp):
  """
  Feature extraction for continuous world.
  """
  # Raw results
  """
  target = {'bias': 0.51638480403475961, 'dist': -0.083742023988640099}
  obstacle = {'bias': -0.91251246907492323, 'dist': 1.9383664807859244}
  segment = {'bias': 0.080048736631393835, 'dist': -0.041394412243896173}
  """

  target = {'bias': 1, 'dist': -0.16}
  obstacle = {'bias': -1, 'dist': 0.16}
  segment = {'bias': 0.1, 'dist': -0.05}
  
  def radiusBias(state, action, label, w):
    extractor = featureExtractors.ContinousRadiusLogExtractor(mdp, label)
    feats = extractor.getFeatures(state, action)

    if feats != None:
      return feats['bias'] * w['bias'] + feats['dist'] * w['dist']
    else:
      return 0

  def qTarget(state, action):
    return radiusBias(state, action, 'targs', target)

  def qObstacle(state, action):
    return radiusBias(state, action, 'obsts', obstacle)

  def qSegment(state, action):
    return radiusBias(state, action, 'segs', segment)

  return [qTarget, qObstacle, qSegment]
