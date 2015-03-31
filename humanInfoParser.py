import humanWorld
import util
from graphics import *

import numpy as np
import continuousWorldDomains
import continuousWorldPlot
import featureExtractors

def plotHuman(plotting, win, subjIdSet, domainId):
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
        
        """
        # a tiny line indicates orient
        arrowLength = 15
        dx = arrowLength * np.cos(orient)
        dy = arrowLength * np.sin(orient)
        arrow = Line(Point(loc), Point(loc[0] + dx, loc[1] + dy))
        arrow.setWidth(2)
        arrow.setFill(color_rgb(100, 100, 100))
        arrow.draw(win)
        """
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
      
      actions = mat['pRes'][idx].action

      # the current segment needs to be read from the domain files
      domain = continuousWorldDomains.loadFromMat("miniRes25.mat", idx)
      # cut the head and tail samples
      for i in range(5, len(targDist) - 15):
        # find the path seg in the mat file
        segList = domain['objs']['segs']
        #print 'agent', (x[i], y[i]), orient[i]
        #print 'parsed seg', segDist[i], segAngle[i]
        for segIdx in xrange(len(segList)):
          dist, angle = featureExtractors.getDistAngle((x[i], y[i]), segList[segIdx], orient[i])
          # right-ward is negative
          angle = -angle
          #print 'seg', segList[segIdx], 'dist', dist, 'angle', angle
          if abs(segDist[i] - dist) < 0.001 and abs(segAngle[i] - angle) < 0.001:
            # then get the one before it or after it
            curSegDistInstance, curSegAngleInstance = featureExtractors.getDistAngle((x[i], y[i]), segList[segIdx - 1], orient[i])
            curSegAngleInstance = -curSegAngleInstance
            break

        if not 'curSegDistInstance' in locals():
          raise Exception('Fail to find the current segment from mat file')

        state = ((targDist[i], targAngle[i]),
                 (targDist2[i], targAngle2[i]),
                 (obstDist[i], obstAngle[i]),
                 (obstDist2[i], obstAngle2[i]),
                 (segDist[i], segAngle[i]),
                 (curSegDistInstance, curSegAngleInstance))
        action = actions[i]
        samples.append((state, action))
        #print (state, action)
        #print featureExtractors.getProjectionToSegmentLocalView(state[4], state[5])
        #raw_input("wait")

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

  for i in range(len(agentXs)):
    samples.append({'x': agentXs[i], 'y': agentZs[i], 'orient': agentAngles[i], 'action': actions[i], 'moveAngle': moveAngles[i]})

  return samples

if __name__ == '__main__':
  if len(sys.argv) < 3:
    print 'args: subjId domainId'
  else:
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
