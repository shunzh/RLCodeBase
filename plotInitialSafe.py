import pickle

import pylab
from numpy import mean

import util
from util import standardErr

rndSeeds = 500

width = height = 5

lensOfQ = {}
times = {}

# will check what methods are run from data
methods = ['iisAndRelpi', 'iisOnly', 'relpiOnly', 'maxProb', 'piHeu']

markers = {'iisAndRelpi': 'bo-', 'iisOnly': 'bs--', 'relpiOnly': 'bd-.', 'maxProb': 'g^-', 'piHeu': 'm+-', 'random': 'c*-'}
names = {'iisAndRelpi': 'SetCover', 'iisOnly': 'SetCover (IIS)', 'relpiOnly': 'SetCover (rel. feat.)', 'maxProb': 'Greed. Prob.',\
         'piHeu': 'Most-Likely', 'random': 'Descending'}

def addFreq(elem, counter): counter[elem] += 1

# output the diffierence of two vectors
vectorDiff = lambda v1, v2: map(lambda e1, e2: e1 - e2, v1, v2)


def plot(x, y, yci, methods, xlabel, ylabel, filename):
  fig = pylab.figure()

  ax = pylab.gca()
  for method in methods:
    print method, y(method), yci(method)
    lines = ax.errorbar(x, y(method), yci(method), fmt=markers[method], mfc='none', label=names[method], markersize=10, capsize=5)

  pylab.xlabel(xlabel)
  pylab.ylabel(ylabel)
  pylab.legend()

  fig.savefig(filename + ".pdf", dpi=300, format="pdf")
  
  pylab.close()


def plotNumVsProprotion():
  """
  Plot the the number of queried features vs the proportion of free features
  """
  #proportionRange = [0.01] + [0.1 * proportionInt for proportionInt in range(10)] + [0.99]
  proportionRange = [0.1, 0.3, 0.5, 0.7, 0.9]
  # fixed carpet num for this exp
  carpets = 10

  for method in methods:
    for proportion in proportionRange:
      lensOfQ[method, proportion] = []
      times[method, proportion] = []

  validInstances = []

  for rnd in range(rndSeeds):
    # set to true if this instance is valid (no safe init policy)
    rndProcessed = False

    for proportion in proportionRange:
      try:
        filename = str(width) + '_' + str(height) + '_' + str(carpets) + '_' + str(proportion) + '_' +  str(rnd) + '.pkl'
        data = pickle.load(open(filename, 'rb'))
      except IOError:
        print filename, 'not exist'
        continue

      # number of features queried
      for method in methods:
        lensOfQ[method, proportion].append(len(data['q'][method]))
        times[method, proportion].append(data['t'][method])
      
      if not rndProcessed:
        rndProcessed = True

        validInstances.append(rnd)
    
  # for output as latex table
  outputFormat = lambda d: '$' + str(round(mean(d), 4)) + ' \pm ' + str(round(standardErr(d), 4)) + '$'

  print 'valid instances', len(validInstances)
  assert len(validInstances) > 0

  # show cases where method1 and method2 are different with proportion. for further debugging methods
  diffInstances = lambda proportion, method1, method2:\
                  (proportion, method1, method2,\
                  filter(lambda _: _[1] != _[2], zip(validInstances, lensOfQ[method1, proportion], lensOfQ[method2, proportion])))

  for proportion in proportionRange: 
    print diffInstances(proportion, 'iisAndRelpi', 'iisOnly')
    print diffInstances(proportion, 'iisAndRelpi', 'relpiOnly')
    print diffInstances(proportion, 'iisAndRelpi', 'maxProb')

  # plot figure
  x = proportionRange
  y = lambda method: [mean(vectorDiff(lensOfQ[method, proportion], lensOfQ['iisAndRelpi', proportion])) for proportion in proportionRange]
  yci = lambda method: [standardErr(vectorDiff(lensOfQ[method, proportion], lensOfQ['iisAndRelpi', proportion])) for proportion in proportionRange]

  plot(x, y, yci, methods, 'Mean of $p_f$', '# of Queried Features', 'lensOfQPf')


def plotNumVsCarpets():
  """
  plot the num of queried features / computation time vs. num of carpets
  """
  carpetNums = [8, 9, 10, 11, 12]

  for method in methods:
    for carpetNum in carpetNums:
      lensOfQ[method, carpetNum] = []
      times[method, carpetNum] = []

  iisSizes = {}
  domPiSizes = {}
  for carpetNum in carpetNums:
    iisSizes[carpetNum] = util.Counter()
    domPiSizes[carpetNum] = util.Counter()

  for rnd in range(rndSeeds):
    for carpetNum in carpetNums:
      try:
        # None as proportion means uniformly random between in 0 and 1
        filename = str(width) + '_' + str(height) + '_' + str(carpetNum) + '_' + 'None' + '_' +  str(rnd) + '.pkl'
        data = pickle.load(open(filename, 'rb'))
      except IOError:
        print filename, 'not exist'
        continue

      # number of features queried
      for method in methods:
        lensOfQ[method, carpetNum].append(len(data['q'][method]))
        times[method, carpetNum].append(data['t'][method])

        addFreq(len(data['iiss']), iisSizes[carpetNum])
        addFreq(len(data['relFeats']), domPiSizes[carpetNum])

  print 'iiss', iisSizes
  print 'relFeats', domPiSizes

  print '# of queries'
  x = carpetNums
  y = lambda method: [mean(vectorDiff(lensOfQ[method, carpetNum], lensOfQ['iisAndRelpi', carpetNum])) for carpetNum in carpetNums]
  yci = lambda method: [standardErr(vectorDiff(lensOfQ[method, carpetNum], lensOfQ['iisAndRelpi', carpetNum])) for carpetNum in carpetNums]
  plot(x, y, yci, methods, '# of Carpets', '# of Queried Features (SetCover as baseline)', 'lensOfQCarpets')

  print 'compute time'
  x = carpetNums
  y = lambda method: [mean(times[method, carpetNum]) for carpetNum in carpetNums]
  yci = lambda method: [standardErr(times[method, carpetNum]) for carpetNum in carpetNums]
  plot(x, y, yci, methods, 'Computation Time (sec.)', '# of Queried Features', 'timesCarpets')


plotNumVsProprotion()
plotNumVsCarpets()