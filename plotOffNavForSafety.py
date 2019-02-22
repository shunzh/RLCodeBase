import pickle
from numpy import mean
from util import standardErr

rndSeeds = 150

lensOfQ = {}
times = {}

methods = ['iisAndRelpi', 'iisOnly', 'relpiOnly', 'maxProb', 'piHeu', 'random']

for method in methods:
  lensOfQ[method] = []
  times[method] = []

for rnd in range(rndSeeds):
  try:
    data = pickle.load(open(str(rnd) + '.pkl'))
  except IOError:
    print rnd, 'not exist'
  
  # number of features queried
  for method in methods:
    lensOfQ[method].append(len(data['q'][method]))
    times[method].append(data['t'][method])

outputFormat = lambda d: '$' + str(round(mean(d), 4)) + ' \pm ' + str(round(standardErr(d), 4)) + '$'

print 'len'
for method in methods:
  print outputFormat(lensOfQ[method])

print 'time'
for method in methods:
  print outputFormat(times[method])