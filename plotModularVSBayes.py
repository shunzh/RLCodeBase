"""
Bar chart demo with pairs of bars grouped for easy comparison.
"""
import numpy as np
import matplotlib.pyplot as plt
import ast
import config
import util

n_groups = 2

def parseFiles(filename, trueW, xAxis, perTrial):
  means = []
  cis = []

  for xId in xrange(len(xAxis)):
    errors = []
    for trialId in xrange(perTrial):
      condorId = xId * perTrial + trialId
      try:
        f = open(filename + "." + str(condorId), 'r')
        w = ast.literal_eval(f.read())
        errors.append(util.getVectorDistance(w, trueW))
      except:
        print "issue in processing ", filename, condorId
    means.append(np.mean(errors))
    cis.append(1.96 * np.std(errors) / np.sqrt(perTrial))
  
  plt.errorbar(xAxis, means, cis)

expectation = [0.16666666666666666,
 0.3333333333333333,
 0.16666666666666666,
 0.3333333333333333]

parseFiles("eval_modular_vs_bayes/grid_modular_out", expectation, config.BUDGET_SIZES, 10)
parseFiles("eval_modular_vs_bayes/grid_bayes_out", expectation, config.BUDGET_SIZES, 10)

plt.xlabel('Number of Samples')
plt.ylabel('L1-normed error of weights')
plt.legend(["Modular IRL", "Bayesian IRL"])
plt.savefig("grid_modular_vs_bayes.eps")
plt.close()