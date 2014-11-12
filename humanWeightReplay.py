import inverseModularRL
import modularAgents
import continuousWorld, humanWorld
import util
import featureExtractors
from graphics import *

import numpy as np

def main(domainId):
  init = continuousWorld.loadFromMat('miniRes25.mat', domainId)
  mdp = humanWorld.HumanWorld(init)

  # plot domain and policy
  dim = 800
  plotting = continuousWorld.Plotting(mdp, dim)
  win = plotting.drawDomain()

  # parse human positions and actions
  humanSamples = parseHumanActions("subj25.parsed.mat", domainId)

  prevLoc = None
  for sample in humanSamples:
    loc = plotting.shift((sample['x'], sample['y']))
    loc = (dim - loc[0], dim - loc[1]) # need 180 degree rotation

    if prevLoc:
      line = Line(Point(prevLoc), Point(loc))
      line.setWidth(5)
      line.setFill(color_rgb(0, 255, 0))
      line.draw(win)
    prevLoc = loc

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
  #main(0)
  #main(8)
  main(16)
  #main(24)

  main(8)
