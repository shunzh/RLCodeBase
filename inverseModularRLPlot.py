import numpy as np

def plotGridOfDiscounters(filenameLookup):
  """
  Plot the obj function value in a heatmap.
  Axises are discounters of two modules.
  """
  import matplotlib.pyplot as plt
  
  stepSize = 2
  data = []
  for i in range(0, 9, stepSize):
    row = []
    for j in range(0, 9, stepSize):
      row.append()
    data.append(row)

  data = np.ma.array(data)
  plt.imshow(data, interpolation='none')
  plt.xticks(range(5), np.arange(0,0.9,0.1 * stepSize))
  plt.yticks(range(5), np.arange(0,0.9,0.1 * stepSize))
  plt.xlabel('Obstacle Module Discounter');
  plt.ylabel('Target Module Discounter');

  plt.jet()
  plt.colorbar()

  plt.savefig(filename)
  
  plt.close()


if __name__ == '__main__':
  def condorOutputFile(taskId, d1, d2, d3):
    d1Str, d2Str, d3Str = map(lambda d: str(int(d * 10)), [d1, d2, d3])
    return "out." + str(taskId) + d1Str + d2Str + d3Str
  
  plotGridOfDiscounters(condorOutputFile)