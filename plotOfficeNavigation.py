import pickle
from numpy import mean, std, sqrt, nan
import numpy
import matplotlib
import pylab
from matplotlib.ticker import FormatStrFormatter

# for ijcai submission
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

markers = {'reallyBrute': 'bo--', 'brute': 'bo-',\
           'alg1': 'gs-',
           'chain': 'rd-', 'naiveChain': 'rd--',\
           'relevantRandom': 'm^-', 'random': 'm^--', 'nq': 'c+-'}

#markerStyle = {'alg1': 's', 'chain': 'd', 'relevantRandom': '^', 'random': '^', 'nq': '+'}
colorMap = {'brute': 'b', 'alg1': 'g', 'chain': 'r', 'naiveChain': 'r', 'relevantRandom': 'm', 'random': 'm', 'nq': 'c'}

legends = {'reallyBrute': 'Brute Force', 'brute': 'Brute Force (rel. feat.)',\
           'alg1': 'MMRQ-k',
           'chain': 'CoA', 'naiveChain': 'Naive CoA',\
           'relevantRandom': 'Random (rel. feat.)', 'random': 'Random', 'nq': 'No Query'}


# shared for all functions
trials = 1500
excluded = set()

# the directory where the experiment results are stored
dataDir = 'experiments/12.12/'

kRange = lambda method: range(1,4) if method == 'brute' else range(1,11)
nRange = [10]

# for brute, only k = 0 .. 3 are available
#methods = ['brute', 'alg1', 'chain', 'relevantRandom', 'random', 'nq']
methods = ['alg1', 'chain', 'relevantRandom', 'random', 'nq']

def excludeFailedExperiments():
  for r in range(trials):
    for n in nRange:
      domainFileName = dataDir + 'domain_' + str(n) + '_' + str(r) + '.pkl'
      ret = pickle.load(open(domainFileName, 'rb'))
      if ret == 'INITIALIZED':
        excluded.add(r)
  
  for n in nRange:
    for mr_type in ['mrk']:
      for method in methods:
        for k in kRange(method):
          for r in range(trials):
            if r not in excluded:
              try:
                ret = pickle.load(open(dataDir + method + '_' + mr_type + '_' + str(k) + '_' + str(n) + '_' + str(r) + '.pkl', 'rb'))
              except IOError:
                print 'not reading', method, k, n, r
                excluded.add(r)

  print 'excluded', excluded

# normalize in terms of regret
def normalize(value, best, worst):
  if best == worst:
    # makes no sense
    return None
  else:
    return (value - best) / (worst - best)

def maximumRegretK():
  mr = {}
  nmr = {} # normalized mr
  dmr = {} # delta mr
  time = {}
  q = {}
  regret = {}
  
  validTrials = trials - len(excluded)

  relPhiNum = {}
  for n in nRange:
    for r in range(trials):
      if r in excluded: continue
      domainFileName = dataDir + 'domain_' + str(n) + '_' + str(r) + '.pkl'
      (relFeats, domPis, domPiTime) = pickle.load(open(domainFileName, 'rb'))
      relPhiNum[n, r] = len(relFeats)
  
  # plot distribution over # of relevant features
  print relPhiNum
  hist(relPhiNum.values(), 'brute', '', '$|\Phi_{rel}|$', 'Frequency', 'numRelPhi')
 
  for n in nRange:
    print n
    for mr_type in ['mrk']:
      #mr_label = '$MR$' if mr_type == 'mr' else '$MR_k$'
      #FIXME call it MR no matter if it's MR or MR_k
      mr_label = '$MR$'
      title = "$|\Phi_?| = " + str(n) + "$"
      for method in methods:
        for k in kRange(method):
          mr[method, k, n, mr_type] = []
          time[method, k, n, mr_type] = []
          q[method, k, n, mr_type] = []
          regret[method, k, n, mr_type] = []
          for r in range(trials):
            if r not in excluded:
              try:
                ret = pickle.load(open(dataDir + method + '_' + mr_type + '_' + str(k) + '_' + str(n) + '_' + str(r) + '.pkl', 'rb'))
                mr[method, k, n, mr_type].append(ret[mr_type])
                time[method, k, n, mr_type].append(ret['time'])
                q[method, k, n, mr_type].append(ret['q'])
                #regret[method, k, n, mr_type].append(ret['regret'])
              except IOError:
                print 'not reading', method, k, n, r

      for method in methods:
        for k in kRange(method):
          nmr[method, k, n, mr_type] = []
          dmr[method, k, n, mr_type] = []
          for r in range(validTrials):
            normalizedmr = normalize(mr[method, k, n, mr_type][r], mr['alg1', k, n, mr_type][r], mr['nq', k, n, mr_type][r])
            if normalizedmr != None:
              nmr[method, k, n, mr_type].append(normalizedmr)
            
            dmr[method, k, n, mr_type].append(mr[method, k, n, mr_type][r] - mr['alg1', k, n, mr_type][r])
            
          hist(dmr[method, k, n, mr_type], method, legends[method] + ", k = " + str(k), "$MR(\Phi_q) - MR(\Phi_q^{MMR})$", "Frequency",
                       "mrkFreq_" + method + "_" + str(n) + "_" + str(k))
        
      """
      print 'measured by mr/mrk'
      plot(kRange, lambda method: [mean(mr[method, _, n, mr_type]) for _ in kRange], lambda method: [standardErr(mr[method, _, n, mr_type]) for _ in kRange],
           methods, title, "k", mr_label, "mr_" + str(n) + "_" + mr_type)

      """
      print 'measured by normalized mr/mrk'
      plot(kRange, lambda method: [mean(nmr[method, _, n, mr_type]) for _ in kRange(method)], lambda method: [standardErr(nmr[method, _, n, mr_type]) for _ in kRange(method)],
           methods, title, "k", "Normalized " + mr_label, "nmr_" + str(n) + "_" + mr_type)
      """

      # COMPARING WITH ALG1 for now
      print 'ratio of finding mmr-q'
      plot(kRange, lambda method: [100.0 * sum(mr[method, k, n, mr_type][_] == mr['alg1', k, n, mr_type][_] for _ in range(validTrials)) / validTrials for k in kRange], lambda _: [0.0] * len(kRange),
           methods, title, "k", "% of Finding a MMR Query", "ratiok_" + str(n) + "_" + mr_type)

      print 'measured by expected regret'
      plot(kRange, lambda method: [mean(regret[method, _, n, mr_type]) for _ in kRange], lambda method: [standardErr(regret[method, _, n, mr_type]) for _ in kRange],
           methods, title, "k", "Expected Regret", "regret_" + str(n) + "_" + mr_type)
      """

      # FIXME may require plotting brute force as well
      print 'time'
      plot(kRange, lambda method: [mean(time[method, _, n, mr_type]) for _ in kRange(method)], lambda method: [standardErr(time[method, _, n, mr_type]) for _ in kRange(method)],
           methods, title, "k", "Computation Time (sec.)", "t_" + str(n) + "_" + mr_type)
  
  assert all(_ >= 0 for _ in regret.values())
  """
  print 'debug'
  validSeq = [_ for _ in range(trials) if _ not in excluded]
  for n in nRange:
    for k in kRange:
      print k, n, [(validSeq[r], relPhiNum[n, r], mr['chain', k, n, 'mrk'][r] - mr['naiveChain', k, n, 'mrk'][r],
                    len(q['chain', k, n, 'mrk'][r]), len(q['naiveChain', k, n, 'mrk'][r]))
                   for r in range(validTrials) if
                   k == relPhiNum[n ,r] and mr['chain', k, n, 'mrk'][r] > mr['naiveChain', k, n, 'mrk'][r]]
  """


def maximumRegretCVSRelPhi():
  mr = {}
  nmr = {} # normalized mr
  time = {}
  kRange = range(1,11) # same for all figures

  validTrials = trials - len(excluded)

  relPhiNum = {}
  for n in nRange:
    for r in range(trials):
      if r in excluded: continue
      domainFileName = dataDir + 'domain_' + str(n) + '_' + str(r) + '.pkl'
      (relFeats, domPis, domPiTime) = pickle.load(open(domainFileName, 'rb'))
      relPhiNum[n, r] = len(relFeats)
  
  #print relPhiNum

  # granularity of x axis
  gran = 1
  bins = range(max(nRange) / gran + 1)

  for k in kRange:
    print k

    for mr_type in ['mrk']:
      #mr_label = '$MR$' if mr_type == 'mr' else '$MR_k$'
      #FIXME call it MR no matter if it's MR or MR_k
      mr_label = '$MR$'
      title = "$k = " + str(k) + "$"
      print mr_type

      xScatter = {}
      yScatter = {}
      for method in methods:
        xScatter[method] = []
        yScatter[method] = []
        for bin in bins: 
          mr[method, k, bin] = []
          time[method, k, bin] = []

        for n in nRange:
          for r in range(trials):
            if r in excluded: continue
            #FIXME weird to read the data file multiple times, but we are representing in a different way. should be fine.
            ret = pickle.load(open(dataDir + method + '_' + mr_type + '_' + str(k) + '_' + str(n) + '_' + str(r) + '.pkl', 'rb'))
            mr[method, k, relPhiNum[n, r] / gran].append(ret[mr_type])
            time[method, k, relPhiNum[n, r] / gran].append(ret['time'])
            
            xScatter[method].append(relPhiNum[n, r])
            yScatter[method].append(ret[mr_type])
            
      for method in methods:
        for bin in bins:
          nmr[method, k, bin] = []
          for r in range(len(mr['alg1', k, bin])):
            normalizedmr = normalize(mr[method, k, bin][r], mr['alg1', k, bin][r], mr['nq', k, bin][r])
            if normalizedmr != None:
              nmr[method, k, bin].append(normalizedmr)

      """
      print 'measured by mr/mrk'
      plot([_ * gran for _ in bins], lambda method: [mean(mr[method, k, _]) for _ in bins], lambda method: [standardErr(mr[method, k, _]) for _ in bins],
           methods, title, "|$\Phi_{rel}$|", mr_label, "mrc_" + str(k) + "_" + mr_type)
      """

      print 'measured by normalized mr/mrk'
      plot([_ * gran for _ in bins], lambda method: [mean(nmr[method, k, _]) for _ in bins], lambda method: [standardErr(nmr[method, k, _]) for _ in bins],
           methods, title, "|$\Phi_{rel}$|", "Normalized " + mr_label, "nmrc_" + str(k) + "_" + mr_type)

      #scatter(xScatter, yScatter, methods, title, "|$\Phi_{rel}$|", "Maximum Regret (" + mr_label + ")", "mrc_" + str(k) + "_" + mr_type)

      """
      print 'time'
      plot([_ * gran for _ in bins], lambda method: [mean(time[method, k, _]) for _ in bins], lambda method: [standardErr(time[method, k, _]) for _ in bins],
           methods, title, "|$\Phi_{rel}$|", "Computation Time (sec.)", "tc_" + str(k) + "_" + mr_type)

      print 'ratio of finding mmr-q'
      plot([_ * gran for _ in bins], lambda method: [100.0 * sum(mr[method, k, bin][_] == mr['alg1', k, bin][_] for _ in range(len(mr['alg1', k, bin]))) / (len(mr['alg1', k, bin])) if len(mr['alg1', k, bin]) > 0 else nan for bin in bins], lambda _: [0.0] * len(bins),
           methods, title, "|$\Phi_{rel}$|", "% of Finding a MMR Query", "ratioc_" + str(k) + "_" + mr_type)
      """


def printTex(y, yci, t, tci, methods, legends):
  for i in range(len(methods)):
    method = methods[i]
    print legends[i], '& $', round(y(method), 4), '\pm', round(yci(method), 4), '$ &',
    print '$', round(t(method), 4), '\pm', round(tci(method), 4), '$ \\\\'

def plot(x, y, yci, methods, title, xlabel, ylabel, filename):
  fig = pylab.figure()

  ax = pylab.gca()
  for method in methods:
    print method, y(method), yci(method)
    lines = ax.errorbar(x(method), y(method), yci(method), fmt=markers[method], label=legends[method], mfc='none', markersize=15, capsize=5)
  
  # easiest way to plot brute force
  method = 'brute'
  lines = ax.errorbar(range(1,4), [0,] * 3, [0,] * 3, fmt=markers[method], label=legends[method], mfc='none', markersize=15, capsize=5)

  ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

  pylab.title(title)
  pylab.xlabel(xlabel)
  pylab.ylabel(ylabel)
  #pylab.ylim([0, 0.2])
  pylab.gcf().subplots_adjust(bottom=0.2, left=0.2)
  fig.savefig(filename + ".pdf", dpi=300, format="pdf")
  
  figLegend = pylab.figure(figsize = (6, 3.7))
  pylab.figlegend(*ax.get_legend_handles_labels(), loc = 'upper left')
  figLegend.savefig("legend.pdf", dpi=300, format="pdf")
  
  pylab.close()

def hist(x, method, title, xlabel, ylabel, filename):
  fig = pylab.figure()

  ax = pylab.gca()
  weights = numpy.ones_like(x)/float(len(x))
  ax.hist(x, color=colorMap[method], weights=weights)

  pylab.title(title)
  pylab.xlabel(xlabel)
  pylab.ylabel(ylabel)
  pylab.ylim([0, 1])
  pylab.gcf().subplots_adjust(bottom=0.2, left=0.2)
  fig.savefig(filename + ".pdf", dpi=300, format="pdf")
  
  pylab.close()

def standardErr(data):
  return std(data) / sqrt(len(data))

if __name__ == '__main__':
  font = {'size': 28}
  matplotlib.rc('font', **font)
  
  #excludeFailedExperiments()
  
  maximumRegretK()

  #maximumRegretCVSRelPhi()
