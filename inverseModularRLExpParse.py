import json
import pickle
import numpy as np
import util

def plotGridOfDiscounters(taskId, getObjValue):
  """
  Plot the obj function value in a heatmap.
  Axises are discounters of two modules.
  """
  import matplotlib.pyplot as plt
  
  stepSize = 1
  data = []
  for i in range(1, 10, stepSize):
    row = []
    for j in range(1, 10, stepSize):
      # add value as a grid
      row.append(getObjValue(taskId, i, j, 6)[0])
    data.append(row)

  data = np.ma.array(data)
  plt.imshow(data, interpolation='none')
  plt.xticks(range(9), np.arange(0.1, 1, 0.1 * stepSize))
  plt.yticks(range(9), np.arange(0.1, 1, 0.1 * stepSize))
  plt.xlabel('Obstacle Module Discounter');
  plt.ylabel('Target Module Discounter');
  plt.title("Objective Function Value with Different Discounters, Task #" + str(taskId + 1))

  plt.jet()
  plt.colorbar()

  plt.savefig('discounterGrid' + str(taskId + 1) + ".png")
  
  plt.close()

def getOptimalWeightDiscounter(taskId, getObjValue):
  """
  Iterate over all out files, find the weight + discounter so that the obj value is minimized
  """
  minObjV = np.inf
  optW = optD = None
  
  #FIXME overfit discounters
  stepSize = 1
  for i in range(1, 10, stepSize):
    for j in range(1, 10, stepSize):
      for k in range(1, 10, stepSize):
        objV, w = getObjValue(taskId, i, j, k)
        
        if objV < minObjV:
          minObjV = objV
          optW = w
          optD = map(lambda x: x * .1, [i, j, k])
  
  return [optW, optD]

def condorObjValue(taskId, d1, d2, d3):
  """
  get obj value given task id and discounters.
  arguments are given in integer formats.
  """
  taskStrLmd = lambda x: str(x) if x > 0 else ''
  filename = "out." + taskStrLmd(taskId) + str(d1) + str(d2) + str(d3)

  with open(filename) as data_file:    
    data = json.load(data_file)
    return data

if __name__ == '__main__':
  # plot
  [plotGridOfDiscounters(taskId, condorObjValue) for taskId in xrange(4)]
  
  # generate value.pkl
  results = []
  for taskId in range(4):
    w, d = getOptimalWeightDiscounter(taskId, condorObjValue)
    results.append(w + d)
  
  util.saveToFile('values.pkl', results)