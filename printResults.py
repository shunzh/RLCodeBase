"""
Print human-readable learning results to std output. 
"""

import pickle
from pprint import pprint
import util

def loadFile(filename):
  try:
    values = pickle.load(open(filename + '.pkl')) or pickle.load(open('learnedValues/' + filename + '.pkl'))
  except:
    print "try to merge condor output files"
    try:
      values = []
      for idx in range(4):
        values.append(pickle.load(open(filename + str(idx) + '.pkl')))
      util.saveToFile(filename + ".pkl", values)
    except:
      print "file not found" + filename
  
  return values or []

taskNames = ['Path only', 'Obstacle + Path', 'Target + Path', 'All']

values = loadFile("values")
evaluations = loadFile("evaluation")

moduleNum = 3
for idx in range(4):
  print taskNames[idx]
  print 'Labels: [target, obstacle, path]'
  print 'Rewards:', [round(x, 3) for x in values[idx][:moduleNum]]
  print 'Discounters:', [round(x, 3) for x in values[idx][moduleNum:2*moduleNum]]
  print 'Others:', [round(x, 3) for x in values[idx][2*moduleNum:]]
  print 'Evaluation:', pprint(evaluations[idx]) # print the dictionary structure of the evaluation results
  print
