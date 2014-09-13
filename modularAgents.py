from qlearningAgents import ApproximateQAgent
from game import Actions
import util

import math

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

    self.qTable = util.Counter()

    self.weights = None
    self.learningWeights = True
 
  def getQValue(self, state, action):
    """
      Get Q value by consulting each module
    """
    return self.qTable[(state, action)]
  
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
  
  def getPolicy(self, state):
    """
      If self.weights
      Can toggle between using QValue directly (traditional way)
      or by proportion of exp(QValue)
    """
    if not self.learningWeights:
      #return ApproximateQAgent.getPolicy(self, state)
      return self.getGibbsPolicy(state)
    else:
      #TODO
      pass
  
  def getGibbsPolicy(self, state):
    """
      Rather than using QValue, use proportion of exp(QValue)
    """
    actions = self.getLegalActions(state)
    if actions: 
      vMat = []
      # iterate through all the modules
      for idx in range(len(self.qFuncs)):
        qFunc = lambda action: self.qFuncs[idx](state, action)

        # list of exp^q
        exps = [math.exp(qFunc(action)) for action in actions]

        # Normalize
        sumExps = sum(exps)
        vMat.append([exp / sumExps for exp in exps])

      values = [(vMat[0][j] * self.weights[0] + vMat[1][j] * self.weights[1]) for j in range(len(actions))]
      for i in range(len(actions)):
        self.qTable[(state, actions[i])] = values[i]

      return actions[values.index(max(values))]
    else:
      return None


def getObsAvoidFuncs(mdp):
  """
    Return Q functiosn for modular mdp for obstacle avoidance behavior

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

  return [qWalk, qObstacle]
  #return [qWalk, qObstacle, qTarget]
