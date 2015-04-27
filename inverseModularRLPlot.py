import json
import numpy as np

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
      row.append(getObjValue(taskId, i, j, 6))
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

def condorObjValue(taskId, d1, d2, d3):
  """
  get obj value given task id and discounters.
  arguments are given in integer formats.
  """
  taskStrLmd = lambda x: str(x) if x > 0 else ''
  filename = "out." + taskStrLmd(taskId) + str(d1) + str(d2) + str(d3)

  with open(filename) as data_file:    
    data = json.load(data_file)
    return data[0]

if __name__ == '__main__':
  [plotGridOfDiscounters(taskId, condorObjValue) for taskId in xrange(4)]