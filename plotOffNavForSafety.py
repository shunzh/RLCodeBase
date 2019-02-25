import pickle

import pylab
from numpy import mean

import util
from util import standardErr

rndSeeds = 1000

width = height = 5
carpets = 10

lensOfQ = {}
times = {}

iisSizes = util.Counter()
minIISSizes = util.Counter()

domPiSizes = util.Counter()
minDomPiSizes = util.Counter()

#proportionRange = [0.01] + [0.1 * proportionInt for proportionInt in range(10)] + [0.99]
proportionRange = [0.1, 0.3, 0.5, 0.7, 0.9]

# will check what methods are run from data
methods = ['iisAndRelpi', 'iisOnly', 'relpiOnly', 'maxProb', 'piHeu', 'random']
for method in methods:
  for proportion in proportionRange:
    lensOfQ[method, proportion] = []
    times[method, proportion] = []

def addFreq(elem, counter):
  counter[elem] += 1

def plot(x, y, yci, methods, xlabel, ylabel, filename):
  fig = pylab.figure()

  ax = pylab.gca()
  for method in methods:
    print method, y(method), yci(method)
    lines = ax.errorbar(x, y(method), yci(method), label=method)

  pylab.xlabel(xlabel)
  pylab.ylabel(ylabel)
  pylab.legend()

  fig.savefig(filename + ".pdf", dpi=300, format="pdf")
  
  pylab.close()

validInstances = []

for rnd in range(rndSeeds):
  # set to true if this instance is valid (no safe init policy)
  valid = False

  for proportion in proportionRange:
    try:
      filename = str(width) + '_' + str(height) + '_' + str(carpets) + '_' + str(proportion) + '_' +  str(rnd) + '.pkl'
      data = pickle.load(open(filename, 'rb'))

      if not valid:
        valid = True
        validInstances.append(rnd)
    except IOError:
      print filename, 'not exist'
      continue

    # number of features queried
    for method in methods:
      lensOfQ[method, proportion].append(len(data['q'][method]))
      times[method, proportion].append(data['t'][method])
    
    """
    # length of iiss
    addFreq(len(data['iiss']), iisSizes)
    # length of the minimum size iis
    addFreq(min(map(lambda s: len(s), data['iiss'])), minIISSizes)
    # length of dom pis
    addFreq(len(data['relFeats']), domPiSizes)
    # length of the dom pi that contains the least number of rel feats
    addFreq(min(map(lambda s: len(s), data['relFeats'])), minDomPiSizes)
    """
  
outputFormat = lambda d: '$' + str(round(mean(d), 4)) + ' \pm ' + str(round(standardErr(d), 4)) + '$'

print 'valid instances', len(validInstances)

assert len(validInstances) > 0

"""
print 'iiss', iisSizes
print 'miniiss', minIISSizes

print 'domPis', domPiSizes
print 'minRelfeats', minDomPiSizes
"""

# show cases where method1 and method2 are different with proportion. for further debugging methods
diffInstances = lambda proportion, method1, method2:\
                (proportion, method1, method2,\
                 filter(lambda _: _[1] != _[2], zip(validInstances, lensOfQ[method1, proportion], lensOfQ[method2, proportion])))

for proportion in proportionRange: 
  print diffInstances(proportion, 'iisAndRelpi', 'iisOnly')
  print diffInstances(proportion, 'iisAndRelpi', 'relpiOnly')
  print diffInstances(proportion, 'iisAndRelpi', 'maxProb')

x = proportionRange
y = lambda method: [mean(lensOfQ[method, proportion]) for proportion in proportionRange]
yci = lambda method: [standardErr(lensOfQ[method, proportion]) for proportion in proportionRange]

plot(x, y, yci, methods, 'Mean of $p_f$', '# of Queried Features', 'lensOfQ')
