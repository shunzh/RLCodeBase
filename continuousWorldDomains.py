import random
import util
import numpy
import warnings

def simpleToyDomain(category = 'targs'):
  """
  This domain can be configed using the category argument,
  so that we only have one target / obstacle for training.

  We put irrelevant objects in infPos that beyond the border,
  so that the agent cannot reach.
  """
  ret = {}

  size = 3.0

  # place that can't be reached
  infPos = (size + 1, size + 1)

  if category == 'targs':
    targs = [(size / 2, size / 2)]; obsts = [infPos]; segs = [infPos]
    # set the starting point to be random for training
    entrance = (random.random() * size, random.random() * size)
  elif category == 'obsts':
    obsts = [(size / 2, size / 2)]; targs = [infPos]; segs = [infPos]
    # random entrance point near the center
    entrance = (size * 2 / 5 + random.random() * size / 5, size * 2 / 5 + random.random() * size / 5)
  elif category == 'segs':
    segs = [(lambda: (random.random() * size, random.random() * size))() for _ in xrange(2)]
    obsts = [infPos]; targs = [infPos]
    entrance = (random.random() * size, random.random() * size)

  elevators = []
  ret['objs'] = {'targs': targs, 'obsts': obsts, 'segs': segs, 'elevators': elevators, 'entrance': entrance}

  ret['xBoundary'] = [0, size]
  ret['yBoundary'] = [0, size]

  ret['radius'] = 0.075
  ret['step'] = 0.1
  
  ret['category'] = category

  return ret

def toyDomain(category = 'targs'):
  """
  Similar to the simple toy domain, but with multiple (same) objects.
  """
  ret = {}

  layout = [(0.3 + 0.4 * x, 0.3 + 0.4 * y) for x in xrange(0, 2) for y in xrange(0, 2) ]
  infPos = (2, 2)

  if category == 'targs':
    targs = layout; obsts = [infPos]
  elif category == 'obsts':
    obsts = layout; targs = [infPos]
  segs = [infPos]

  elevators = [(1, 1)]
  entrance = (0, 0)
  ret['objs'] = {'targs': targs, 'obsts': obsts, 'segs': segs, 'elevators': elevators, 'entrance': entrance}

  ret['xBoundary'] = [-0.1, 1.1]
  ret['yBoundary'] = [-0.1, 1.1]

  # radius of an object (so the object doesn't appear as a point)
  ret['radius'] = 0.075

  # step size of the agent movement
  ret['step'] = 0.1

  ret['category'] = category

  return ret

def loadFromMat(filename, domainId, randInit = False):
  """
  Load from mat file that provided by Matt.

  Args:
    filename: name of the mat file, presumably in the same directory.
    domainId: there should be multiple configurations of rooms in this file,
              indicate which room to use.
  Return:
    A dictionary with necessary keys.
  """
  # read layout from source file
  s = util.loadmat(filename)

  ret = {}

  numObj = len(s['newRes']['all_objs']['id'])

  targs = []
  obsts = []
  segs = []
  elevators = []

  for idx in xrange(numObj):
    name = s['newRes']['all_objs']['id'][idx]

    x = s['newRes']['all_objs']['object_location']['x'][domainId][idx]
    y = s['newRes']['all_objs']['object_location']['z'][domainId][idx]

    if 'targ' in name:
      targs.append((x, y))
    elif 'obst' in name:
      obsts.append((x, y))
    elif 'seg' in name:
      # FIXME just guesses
      # drop far located segments
      if numpy.linalg.norm((x, y)) < 10:
        segs.append((x, y))
    elif 'elevator' in name:
      elevators.append((x, y))
    else:
      warnings.warn("Dropped unkown object typed '" + name + "' indexed at " + str(idx))

  # FIXME segs need to be reversed for even numbered rooms
  if domainId % 2 == 0:
    segs.reverse()

  if len(elevators) == 0:
    raise Exception("Elevators cannot be undefined.")

  # entrance is always the position symmetric to the elevator wrt the origin
  entrance = (-elevators[0][0], -elevators[0][1])

  ret['objs'] = {'targs': targs, 'obsts': obsts, 'segs': segs, 'elevators': elevators, 'entrance': entrance}

  maxCor = 4
  ret['xBoundary'] = [-maxCor, maxCor]
  ret['yBoundary'] = [-maxCor, maxCor]

  if randInit:
    x = ret['xBoundary'][0] + random.random() * (ret['xBoundary'][1] - ret['xBoundary'][0])
    y = ret['yBoundary'][0] + random.random() * (ret['yBoundary'][1] - ret['yBoundary'][0])
    ret['objs']['elevators'][0] = (x, y)

  # FIXME overfit
  # radius of an object (so the object doesn't appear as a point)
  ret['radius'] = 0.1905
  # step size of the agent movement
  ret['step'] = 0.3

  return ret
