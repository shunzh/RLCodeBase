"""
Print human-readable learning results to std output. 
"""

import pickle

values = pickle.load(open('learnedValues/values.pkl'))
agreedPolicies = pickle.load(open('agreedPolicies.pkl'))
taskNames = ['Path only', 'Obstacle + Path', 'Target + Path', 'All']

print "The values for the four tasks are, given in the form of [target, obstacle, path]:"
print

moduleNum = 3
for idx in range(4):
  print taskNames[idx]
  print 'Weights:', [round(x, 4) for x in values[idx][:moduleNum]]
  print 'Discounters:', [round(x, 4) for x in values[idx][moduleNum:]]
  print 'Proportion of agreed policies:', "%.4f" % agreedPolicies[idx]
  print
