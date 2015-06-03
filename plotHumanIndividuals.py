import ast
import numpy as np
import matplotlib.pyplot as plt

def parseFiles(filename, rang):
  xs = []
  for condorId in rang:
    with open(filename + "." + str(condorId), 'r') as f:
      for line in f:
        pass
      last = line
      xs.append(ast.literal_eval(last))

  # column wise
  means = np.mean(xs, 0)
  cis = 1.96 * np.std(xs, 0) / np.sqrt(len(rang))
  
  return (means, cis)

perHuman = 32
perTask = 8

def dataRange(humanId, taskId):
  start = perHuman * humanId + perTask * taskId
  return range(start, start + perTask)

def getHumanResults(rang):
  """
  return [[mean of x], [ci of x]] 
  """
  return parseFiles("eval_human_individuals/out", rang)

barwidth = 0.2
colors = ['r', 'y', 'b', 'g']
error_config = {'ecolor': '0.3'}
legends = ["Subject #1", "Subject #2", "Subject #3", "Subject #4"]
def plotBars(index, means, cis, lbl):
  # FIXME no other solution found
  i = 0
  plt.bar(i + barwidth * index, means[i], barwidth,
          yerr=cis[i],
          color=colors[index],
          label=legends[index],
          error_kw=error_config)
  i = 1
  plt.bar(i + barwidth * index, means[i], barwidth,
          yerr=cis[i],
          color=colors[index],
          error_kw=error_config)
  i = 2
  plt.bar(i + barwidth * index, means[i], barwidth,
          yerr=cis[i],
          color=colors[index],
          error_kw=error_config)

def plotDiscounterBars(index, means, cis, lbl):
  # FIXME no other solution found
  i = 3
  plt.bar(i - 3 + barwidth * index, means[i], barwidth,
          yerr=cis[i],
          color=colors[index],
          label=legends[index],
          error_kw=error_config)
  i = 4
  plt.bar(i - 3 + barwidth * index, means[i], barwidth,
          yerr=cis[i],
          color=colors[index],
          error_kw=error_config)
  i = 5
  plt.bar(i - 3 + barwidth * index, means[i], barwidth,
          yerr=cis[i],
          color=colors[index],
          error_kw=error_config)

data = [getHumanResults(rang) for rang in [dataRange(0, 3), dataRange(1, 3), dataRange(2, 3), dataRange(3, 3)]] 
print data

def plot(pFun, ylabel, filename):
  for i in xrange(4):
    pFun(i, data[i][0], data[i][1], "Human #" + str(i))

  plt.gcf().set_size_inches(5,4)
  plt.ylabel(ylabel)
  plt.xticks(np.arange(3) + 0.5, ('Target', 'Obstacle', 'Path'))
  plt.legend(loc='upper left')

  plt.savefig(filename + ".png")
  plt.close()
  
plot(plotBars, "Weight", "human_individuals_weights")
plot(plotDiscounterBars, "Discount Factor", "human_individuals_discounters")