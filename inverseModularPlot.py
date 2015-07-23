import numpy as np

def printFitness(sln, filename):
  import matplotlib.pyplot as plt

  stepSize = 1
  data = []
  for i in range(0, 10, stepSize):
    row = []
    for j in range(0, 10, stepSize):
      value = - sln.obj([1, 1, .1 * i, .1 * j])
      row.append(value)
    data.append(row)

  plt.imshow(data, interpolation='none')
  plt.xticks(range(10), np.arange(0,1,0.1 * stepSize))
  plt.yticks(range(10), np.arange(0,1,0.1 * stepSize))
  plt.xlabel("Module #2 Discount Factor")
  plt.ylabel("Module #1 Discount Factor")

  plt.jet()
  plt.colorbar()
  plt.bone()
  plt.show()

  plt.savefig(filename)
  
  plt.close()

