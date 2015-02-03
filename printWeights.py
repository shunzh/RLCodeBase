import pickle

weights = pickle.load(open('learnedValues/weights.pkl'))
taskNames = ['Path only', 'Obstacle + Path', 'Target + Path', 'All']

print "The weights for the four tasks are:"
print

for idx in range(4):
  print taskNames[idx]
  print weights[idx]
