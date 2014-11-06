import inverseModularRL
import modularAgents
import continuousWorld, humanWorld
import util
import featureExtractors
from graphics import *

import numpy as np

def main(domainFrom, domainTo, domainDemo = None):
  domainDemo = domainDemo or domainFrom

  [w, sln] = inverseModularRL.humanWorldExperiment("subj25.parsed.mat", range(domainFrom, domainTo))

  init = continuousWorld.loadFromMat('miniRes25.mat', domainDemo)
  mdp = humanWorld.HumanWorld(init)

  qLearnOpts = {'gamma': 0.9,
                'alpha': 0.5,
                'epsilon': 0, # make sure no exploration in decision making.
                'actionFn': mdp.getPossibleActions}
  qFuncs = modularAgents.getHumanWorldContinuousFuncs()
  aHat = modularAgents.ModularAgent(**qLearnOpts)
  aHat.setQFuncs(qFuncs)
  aHat.setWeights(w) # get the weights in the result

  # plot domain and policy
  dim = 800
  win = continuousWorld.drawDomain(mdp, dim)
  size = max(mdp.xBoundary[1] - mdp.xBoundary[0], mdp.yBoundary[1] - mdp.yBoundary[0])

  # parse human positions and actions
  humanSamples = parseHumanActions("subj25.parsed.mat", domainDemo)
  # use IRL code to get features
  featureSamples = inverseModularRL.getSamplesFromMat("subj25.parsed.mat", [domainDemo])

  def shift(loc):
    """
    shift to the scale of the GraphWin
    """
    return (1.0 * (mdp.xBoundary[1] - loc[0]) / size * dim, 1.0 * (mdp.yBoundary[1] - loc[1]) / size * dim)
 
  def plotCircle(win, loc, color):
    cir = Circle(Point(shift(loc)), 4)
    cir.setFill(color)
    cir.draw(win)

  def plotArrow(win, loc, orient, orientDiff, color):
    orient = - orient + np.pi / 2 - orientDiff
    orient = featureExtractors.adjustAngle(orient)

    segLen = 0.2
    newLoc = (loc[0] + segLen * np.cos(orient), loc[1] + segLen * np.sin(orient))

    line = Line(Point(shift(loc)), Point(shift(newLoc)))
    line.setWidth(2)
    line.setFill(color)
    line.draw(win)

  for i in range(len(featureSamples)):
    # Note that aHat is a Modular agent, not a ReducedModular one. It accepts belief state directly.
    # For estimated agent, it only predicts discrete actions.
    # For human, just use the angle it moves.
    estimatedPolicy = aHat.getPolicy(featureSamples[i][0])
    if estimatedPolicy == 'L':
      estimatedMoveAngle = -mdp.turnAngle
    elif estimatedPolicy == 'R':
      estimatedMoveAngle = mdp.turnAngle
    elif estimatedPolicy == 'G':
      estimatedMoveAngle = 0

    trueMoveAngle = humanSamples[i]['moveAngle']

    loc = (humanSamples[i]['x'], humanSamples[i]['y'])
    orient = humanSamples[i]['orient']

    plotCircle(win, loc, 'white')
    plotArrow(win, loc, orient, estimatedMoveAngle, 'green')
    plotArrow(win, loc, orient, trueMoveAngle, 'white')

  win.getMouse()
  win.close()

def parseHumanActions(filename, domainId):
  """
  Parse human behavivors from mat, which includes at which point, take which action.

  Args:
    filename: mat file to read
    domainId: room number

  Return:
    a list of dicts, which includes x, y, orient and action.
  """
  mat = util.loadmat(filename)

  samples = []

  agentXs = mat['pRes'][domainId].agentX
  agentZs= mat['pRes'][domainId].agentZ
  agentAngles = mat['pRes'][domainId].agentAngle / 180 * np.pi
  moveAngles = mat['pRes'][domainId].agentMoveAngle / 180 * np.pi
  actions = mat['pRes'][domainId].action

  for i in range(len(agentXs)):
    samples.append({'x': agentXs[i], 'y': agentZs[i], 'orient': agentAngles[i], 'action': actions[i], 'moveAngle': moveAngles[i]})

  return samples

if __name__ == '__main__':
  #main(0, 8)
  #main(24, 31)
  #main(8, 16)
  main(16, 24, 17)
