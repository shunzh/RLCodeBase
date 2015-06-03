import ast
import numpy as np

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

def printResults(label, rang):
  data = parseFiles("eval_human_individuals/out", rang)
  print label,
  for mean, ci in np.transpose(data):
    print "&", round(mean, 2), "$\\pm$", round(ci, 2),
  print "\\\\"

printResults("Human Subject #1", dataRange(0, 3))
printResults("Human Subject #2", dataRange(1, 3))
printResults("Human Subject #3", dataRange(2, 3))
printResults("Human Subject #4", dataRange(3, 3))