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
    humanSamples = parseHumanData(['subj' + str(subjId) + '.parsed.mat'], [int(domainId)])

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

def getHumanStatesActions(filenames, idxSet, parsedHumanData = None, getRawAngle = False):
  """
  return (state, action) paris
  """
  parsedHumanData = parsedHumanData or parseHumanData(filenames, idxSet)
  return map(lambda x:(((x['targDist'][0], x['targAngle'][0]),
                        (x['targDist'][1], x['targAngle'][1]),
                        (x['obstDist'][0], x['obstAngle'][0]),
                        (x['obstDist'][1], x['obstAngle'][1]),
                        (x['segDist'][0], x['segAngle'][0]),
                        (x['segDist'][1], x['segAngle'][1])
                       ),
                       x['moveAngle'] if getRawAngle else x['action']),
                       parsedHumanData
            )

def parseHumanData(filenames, domainIds):
  """
  Parse human behavivors from mat, which contains positions, actions and angle changes, etc. at all data points.
  (Action is redundant given change of angles. Just for convenience.)

  Args:
    filenames: mat files to read
    domainIds: room numbers

  Return:
    a list of dicts that contain all possibly useful information
  """
  samples = []

  for filename in filenames:
    mat = util.loadmat(filename)
    
    for domainId in domainIds:
      agentXs = mat['pRes'][domainId].agentX
      agentZs = mat['pRes'][domainId].agentZ
      agentAngles = - mat['pRes'][domainId].agentAngle / 180 * np.pi - np.pi  / 2
      moveAngles = mat['pRes'][domainId].agentMoveAngle / 180 * np.pi

      obstDist = mat['pRes'][domainId].obstDist1
      obstAngle = mat['pRes'][domainId].obstAngle1 / 180.0 * np.pi
      obstDist2 = mat['pRes'][domainId].obstDist2
      obstAngle2 = mat['pRes'][domainId].obstAngle2 / 180.0 * np.pi

      targDist = mat['pRes'][domainId].targDist1
      targAngle = mat['pRes'][domainId].targAngle1 / 180.0 * np.pi
      targDist2 = mat['pRes'][domainId].targDist2
      targAngle2 = mat['pRes'][domainId].targAngle2 / 180.0 * np.pi

      # the next segment, already provided.
      segDist = mat['pRes'][domainId].pathDist
      segAngle = mat['pRes'][domainId].pathAngle / 180.0 * np.pi
 
      for i in range(HEAD_CUTOFF, len(agentXs) - TAIL_CUTOFF):
        if config.TWO_OBJECTS or humanWorldExperiment.nModules == 4:
          # coordinate problems, need negate x, z
          (curSegDistInstance, curSegAngleInstance) = continuousWorldDomains.getPreviousWaypoint\
                                                      (domainId, -agentXs[i], -agentZs[i], agentAngles[i], segDist[i], segAngle[i])
        else:
          # this field will be dummy
          (curSegAngleInstance, curSegDistInstance) = (segDist[i], segAngle[i])

        action = humanWorld.HumanWorld.actions.angleToAction(moveAngles[i])
        samples.append({'x': agentXs[i],\
                        'y': agentZs[i],\
                        'orient': agentAngles[i],\
                        'obstDist': [obstDist[i], obstDist2[i]],\
                        'obstAngle': [obstAngle[i], obstAngle2[i]],\
                        'targDist': [targDist[i], targDist2[i]],\
                        'targAngle': [targAngle[i], targAngle2[i]],\
                        'segDist': [segDist[i], curSegDistInstance],\
                        'segAngle': [segAngle[i], curSegAngleInstance],\
                        'action': action,\
                        'moveAngle': moveAngles[i]})

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
