import humanWorld
import util
import sys

import numpy as np
import continuousWorldDomains
import featureExtractors
import config
import humanWorldExperiment

HEAD_CUTOFF = 5
TAIL_CUTOFF = 15

def plotHuman(plotting, win, subjIdSet, domainId):
  from graphics import Line, Point, color_rgb
  dim = plotting.dim

  # parse human positions and actions
  for subjId in subjIdSet:
    humanSamples = parseHumanActions('subj' + str(subjId) + '.parsed.mat', int(domainId))

    prevLoc = None
    for sample in humanSamples:
      loc = plotting.shift((sample['x'], sample['y']))
      orient = sample['orient']
      loc = (dim - loc[0], dim - loc[1]) # need 180 degree rotation

      if prevLoc:
        # a solid line indicates trajectory
        line = Line(Point(prevLoc), Point(loc))
        line.setWidth(5)
        line.setFill(color_rgb(0, 0, 0))
        line.draw(win)
        
        if config.DEBUG:
          # a tiny line indicates orient
          arrowLength = 15
          dx = arrowLength * np.cos(orient)
          dy = arrowLength * np.sin(orient)
          arrow = Line(Point(loc), Point(loc[0] + dx, loc[1] + dy))
          arrow.setWidth(2)
          arrow.setFill(color_rgb(100, 100, 100))
          arrow.draw(win)
      prevLoc = loc
    prevLoc = None

def getHumanStatesActions(filenames, idxSet):
  """
  Get human actions and states from mat files.
  """
  samples = []

  for filename in filenames:
    mat = util.loadmat(filename)

    for idx in idxSet:
      x = - mat['pRes'][idx].agentX
      y = - mat['pRes'][idx].agentZ
      orient = [featureExtractors.adjustAngle(- angle / 180.0 * np.pi - np.pi / 2) for angle in mat['pRes'][idx].agentAngle] 

      obstDist = mat['pRes'][idx].obstDist1
      obstAngle = mat['pRes'][idx].obstAngle1 / 180.0 * np.pi
      obstDist2 = mat['pRes'][idx].obstDist2
      obstAngle2 = mat['pRes'][idx].obstAngle2 / 180.0 * np.pi

      targDist = mat['pRes'][idx].targDist1
      targAngle = mat['pRes'][idx].targAngle1 / 180.0 * np.pi
      targDist2 = mat['pRes'][idx].targDist2
      targAngle2 = mat['pRes'][idx].targAngle2 / 180.0 * np.pi

      # the next segment, already provided.
      segDist = mat['pRes'][idx].pathDist
      segAngle = mat['pRes'][idx].pathAngle / 180.0 * np.pi
      
      # figure out its action
      moveAngle = mat['pRes'][idx].agentMoveAngle / 180.0 * np.pi

      # cut the head and tail samples
      for i in range(HEAD_CUTOFF, len(targDist) - TAIL_CUTOFF):
        if config.TWO_OBJECTS or humanWorldExperiment.nModules == 4:
          (curSegDistInstance, curSegAngleInstance) = continuousWorldDomains.getPreviousWaypoint\
                                                      (idx, x[i], y[i], orient[i], segDist[i], segAngle[i])
        else:
          # this field will be dummy
          (curSegAngleInstance, curSegDistInstance) = (segDist[i], segAngle[i])

        state = ((targDist[i], targAngle[i]),
                 (targDist2[i], targAngle2[i]),
                 (obstDist[i], obstAngle[i]),
                 (obstDist2[i], obstAngle2[i]),
                 (segDist[i], segAngle[i]),
                 (curSegDistInstance, curSegAngleInstance))
        action = humanWorld.HumanWorld.actions.angleToAction(moveAngle[i])
        
        samples.append((state, action))

  return samples

def parseHumanActions(filename, domainId):
  """
  Parse human behavivors from mat, which contains positions, actions and angle changes at all data points.
  (Action is redundant given change of angles. Just for convenience.)

  Args:
    filename: mat file to read
    domainId: room number

  Return:
    a list of dicts, which includes x, y, orient and action.
  """
  mat = util.loadmat(filename)

  samples = []

  agentXs = mat['pRes'][domainId].agentX
  agentZs = mat['pRes'][domainId].agentZ
  agentAngles = - mat['pRes'][domainId].agentAngle / 180 * np.pi - np.pi  / 2
  moveAngles = mat['pRes'][domainId].agentMoveAngle / 180 * np.pi
  actions = mat['pRes'][domainId].action

  for i in range(HEAD_CUTOFF, len(agentXs) - TAIL_CUTOFF):
    samples.append({'x': agentXs[i], 'y': agentZs[i], 'orient': agentAngles[i], 'action': actions[i], 'moveAngle': moveAngles[i]})

  return samples

if __name__ == '__main__':
  if len(sys.argv) < 3:
    print 'args: subjId domainId'
  else:
    import continuousWorldPlot
    subjId = sys.argv[1]
    domainId = sys.argv[2]

    # create plotting canvas, load room configuration
    init = continuousWorldDomains.loadFromMat('miniRes25.mat', int(domainId))
    mdp = humanWorld.HumanWorld(init)
    dim = 800
    plotting = continuousWorldPlot.Plotting(mdp, dim)
    win = plotting.drawDomain()

    plotHuman(plotting, win, [subjId], domainId)

    win.getMouse()
    win.close()
