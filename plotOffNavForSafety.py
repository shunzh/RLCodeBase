import pickle
from numpy import mean
from util import standardErr
import util

rndSeeds = 300

lensOfQ = {}
times = {}

iisSizes = util.Counter()
minIISSizes = util.Counter()

domPiSizes = util.Counter()
minDomPiSizes = util.Counter()

def addFreq(elem, counter):
  counter[elem] += 1

#methods = ['iisAndRelpi', 'iisOnly', 'relpiOnly', 'maxProb', 'piHeu', 'random']
methods = ['iisAndRelpi', 'iisOnly', 'relpiOnly']

for method in methods:
  lensOfQ[method] = []
  times[method] = []

validInstances = 0

for rnd in range(rndSeeds):
  try:
    data = pickle.load(open(str(rnd) + '.pkl'))
  except IOError:
    print rnd, 'not exist'
  
  validInstances += 1
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

print 'valid instances', validInstances

print 'iiss', iisSizes
print 'miniiss', minIISSizes

print 'domPis', domPiSizes
print 'minRelfeats', minDomPiSizes

print 'len'
for method in methods:
  print outputFormat(lensOfQ[method])

print 'time'
for method in methods:
  print outputFormat(times[method])