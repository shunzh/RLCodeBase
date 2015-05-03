"""
Print human-readable learning results to std output. 
"""

import pickle
from pprint import pprint
import os

if os.path.isfile('values.pkl'):
  values = pickle.load(open('values.pkl'))
else:
  values = pickle.load(open('learnedValues/values.pkl'))

try:
  evaluation = pickle.load(open('evaluation.pkl'))
  evaluationExists = True
except:
  evaluationExists = False

taskNames = ['Path only', 'Obstacle + Path', 'Target + Path', 'All']

print

moduleNum = 3
for idx in range(4):
  print taskNames[idx]
  print 'Labels: [target, obstacle, path]'
  print 'Weights:', [round(x, 4) for x in values[idx][:moduleNum]]
  print 'Discounters:', [round(x, 4) for x in values[idx][moduleNum:]]
  print 'Evaluation:'
  if evaluationExists: pprint(evaluation[idx]) # print the dictionary structure of the evaluation results
  print