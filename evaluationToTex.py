import pickle

table = pickle.load(open('evaluation.pkl'))

agents = ['ModularAgent', 'ReflexAgent', 'RandomAgent']
metrics = ['angularDifference', 'likelihood']
taskNames = ['Path only', 'Obstacle + Path', 'Target + Path', 'All']

for idx in xrange(len(table)):
  print '\multirow{' + str(len(agents)) + '}{*}{' + taskNames[idx] + '}'
  for agent in agents:
    print '& ' + agent,
    for metric in metrics:
      print ' & ' + str(round(table[idx][agent][metric], 4)),
    print '\\\\'
  print '\\hline'