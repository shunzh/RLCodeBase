import pickle

weights = pickle.load(open('learnedValues/weights.pkl'))
taskNames = ['Path only', 'Obstacle + Path', 'Target + Path', 'All']

print "The weights for the four tasks are, given in the form of [target, obstacle, path]:"
print

for idx in range(4):
  print taskNames[idx]
  print weights[idx]
