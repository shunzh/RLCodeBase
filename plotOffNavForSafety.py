import pickle
from numpy import mean
from util import standardErr
import util

rndSeeds = 1000

width = height = 3
carpets = 6

lensOfQ = {}
times = {}

iisSizes = util.Counter()
minIISSizes = util.Counter()

domPiSizes = util.Counter()
minDomPiSizes = util.Counter()

# will check what methods are run from data
methods = ['iisAndRelpi', 'iisOnly', 'relpiOnly', 'maxProb', 'piHeu', 'random']
for method in methods:
  lensOfQ[method] = []
  times[method] = []

def addFreq(elem, counter):
  counter[elem] += 1

validInstances = []

for rnd in range(rndSeeds):
  try:
    filename = str(width) + '_' + str(height) + '_' + str(carpets) + '_' +  str(rnd) + '.pkl'
    data = pickle.load(open(filename, 'rb'))
  except IOError:
    print rnd, 'not exist'
    continue

  # keep track of the random seeds that no initial safe policies exist
  validInstances.append(rnd)

  # number of features queried
  for method in methods:
    lensOfQ[method].append(len(data['q'][method]))
    times[method].append(data['t'][method])
  
  # record the following
  # length of iiss
  addFreq(len(data['iiss']), iisSizes)
  # length of the minimum size iis
  addFreq(min(map(lambda s: len(s), data['iiss'])), minIISSizes)
  # length of dom pis
  addFreq(len(data['relFeats']), domPiSizes)
  # length of the dom pi that contains the least number of rel feats
  addFreq(min(map(lambda s: len(s), data['relFeats'])), minDomPiSizes)

outputFormat = lambda d: '$' + str(round(mean(d), 4)) + ' \pm ' + str(round(standardErr(d), 4)) + '$'

print 'valid instances', len(validInstances)

assert len(validInstances) > 0

print 'iiss', iisSizes
print 'miniiss', minIISSizes

print 'domPis', domPiSizes
print 'minRelfeats', minDomPiSizes

# interesting in the case where variations alg 1 finds different queries
print 'vs iisOnly', filter(lambda _: _[1] != _[2], zip(validInstances, lensOfQ['iisAndRelpi'], lensOfQ['iisOnly']))
print 'vs relpiOnly', filter(lambda _: _[1] != _[2], zip(validInstances, lensOfQ['iisAndRelpi'], lensOfQ['relpiOnly']))

print 'len'
for method in methods:
  print method, outputFormat(lensOfQ[method])

print 'time'
for method in methods:
  print method, outputFormat(times[method])
