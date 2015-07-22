import numpy as np

def printFitness(sln, filename):
  import matplotlib.pyplot as plt

  stepSize = 1
  data = []
  for i in range(0, 11, stepSize):
    row = []
    for j in range(0, 11, stepSize):
      value = sln.obj([1, 1, .1 * i, .1 * j])
      row.append(value)
    data.append(row)

  plt.imshow(data, interpolation='none')
  plt.xticks(range(11), np.arange(0,1.1,0.1 * stepSize))
  plt.yticks(range(11), np.arange(0,1.1,0.1 * stepSize))

  plt.jet()
  plt.colorbar()

  plt.savefig(filename)
  
  plt.close()

