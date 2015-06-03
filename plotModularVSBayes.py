"""
Bar chart demo with pairs of bars grouped for easy comparison.
"""
import numpy as np
import matplotlib.pyplot as plt
import ast
import config

n_groups = 2

def parseFiles(filename, xAxis, perTrial):
  means = []
  cis = []

  for xId in xrange(len(xAxis)):
    errors = []
    for trialId in xrange(perTrial):
      condorId = xId * perTrial + trialId
      try:
        with open(filename + "." + str(condorId), 'r') as f:
          mse = ast.literal_eval(f.read())
          errors.append(mse)
      except:
        print "issue in processing ", filename, condorId
    means.append(np.mean(errors))
    cis.append(1.96 * np.std(errors) / np.sqrt(len(errors)))
    
    print errors
  
  plt.errorbar(xAxis, means, cis)

parseFiles("eval_modular_vs_bayes/grid_modular_out", config.BUDGET_SIZES, 10)
parseFiles("eval_modular_vs_bayes/grid_bayes_out", config.BUDGET_SIZES, 10)

plt.xlabel('Number of Samples')
plt.ylabel('L1-normed error of weights')
plt.legend(["Modular IRL", "Bayesian IRL"])
plt.savefig("grid_modular_vs_bayes.eps")
plt.close()