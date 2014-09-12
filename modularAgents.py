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

  def qObstacle(state, action):
    """
      QValue of obstacle avoiding
    """
    x, y = state
    next_x, next_y = getNext(state, action)

    # find the distance to the nearest obstacle
    minDist = mdp.grid.width * mdp.grid.height
    for xt in range(mdp.grid.width):
      for yt in range(mdp.grid.height):
        cell = mdp.grid[xt][yt] 
        if (type(cell) == int or type(cell) == float) and cell < 0:
          # it's an obstacle!
          dist = math.sqrt((xt - next_x) ** 2 + (yt - next_y) ** 2)
          if (dist < minDist): minDist = dist
    return minDist * obstacle['dis'] + 1 * obstacle['bias']

  return [qWalk, qObstacle]
