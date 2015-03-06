"""
Print human-readable learning results to std output. 
"""

import pickle

weights = pickle.load(open('learnedValues/weights.pkl'))
agreedPolicies = pickle.load(open('agreedPolicies.pkl'))
taskNames = ['Path only', 'Obstacle + Path', 'Target + Path', 'All']

print "The weights for the four tasks are, given in the form of [target, obstacle, path]:"
print

for idx in range(4):
  print taskNames[idx]
  print 'Weights:', [round(x, 4) for x in weights[idx]]
  print 'Proportion of agreed policies:', "%.4f" % agreedPolicies[idx]
  print
