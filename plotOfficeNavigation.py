import pickle
from numpy import mean, std, sqrt, nan
import matplotlib
import pylab
from matplotlib.ticker import FormatStrFormatter

markers = {'reallyBrute': 'bo--', 'brute': 'bo-',\
           'alg1': 'gs-', 'chain': 'rd-',\
           'alg1mr': 'gs-', 'chainmr': 'rd-',\
           'alg1mrk': 'gs--', 'chainmrk': 'rd--',\
           'relevantRandom': 'm^-', 'random': 'm^--', 'nq': 'c+-'}

markerStyle = {'alg1': 's', 'chain': 'd', 'relevantRandom': '^', 'random': '^', 'nq': '+'}
colorMap = {'alg1': 'g', 'chain': 'r', 'relevantRandom': 'm', 'random': 'm', 'nq': 'c'}

legends = {'reallyBrute': 'Brute Force', 'brute': 'Brute Force (rel. feat.)',\
           'alg1': 'MMRQ-k', 'chain': 'CoA',\
           'alg1mr': 'MMRQ-k ($MR$)', 'chainmr': 'CoA ($MR$)',\
           'alg1mrk': 'MMRQ-k ($MR_k$)', 'chainmrk': 'CoA ($MR_k$)',\
           'relevantRandom': 'Random (rel. feat.)', 'random': 'Random', 'nq': 'No Query'}


# shared for all functions
trials = 300
excluded = set()

#methods = ['brute', 'alg1', 'chain', 'relevantRandom', 'random', 'nq']
methods = ['alg1', 'chain']
kRange = [1, 2, 3]
nRange = [10, 15]

def excludeFailedExperiments():
  for r in range(trials):
    for n in nRange:
      domainFileName = 'domain_' + str(n) + '_' + str(r) + '.pkl'
      ret = pickle.load(open(domainFileName, 'rb'))
      if ret == 'INITIALIZED':
        excluded.add(r)
  
  excluded.update([30, 36, 184])
  print 'excluded', excluded

def maximumRegretK():
  mr = {}
  time = {}
  q = {}
  
  validTrials = trials - len(excluded)

  for n in nRange:
    print n
    for mr_type in ['mrk']:
      mr_label = '$MR$' if mr_type == 'mr' else '$MR_k$'
      title = "Objective is " + mr_label + ", $|\Phi_?| = " + str(n) + "$"
      for k in kRange:
        for method in methods:
          # queries that optimize mr_type and measured by mr
          mr[method, k, n, mr_type] = []
          # queries that optimize mr_type and measured by mrk
          time[method, k, n, mr_type] = []
          q[method, k, n, mr_type] = []
          for r in range(trials):
            if r not in excluded:
              ret = pickle.load(open(method + '_' + mr_type + '_' + str(k) + '_' + str(n) + '_' + str(r) + '.pkl', 'rb'))
              mr[method, k, n, mr_type].append(ret[mr_type])
              time[method, k, n, mr_type].append(ret['time'])
              q[method, k, n, mr_type].append(ret['q'])
      
      print 'measured by mr/mrk'
      plot(kRange, lambda method: [mean(mr[method, _, n, mr_type]) for _ in kRange], lambda method: [standardErr(mr[method, _, n, mr_type]) for _ in kRange],
           methods, title, "k", "Maximum Regret (" + mr_label + ")", "mr_" + str(n) + "_" + mr_type)

      print 'time'
      plot(kRange, lambda method: [mean(time[method, _, n, mr_type]) for _ in kRange], lambda method: [standardErr(time[method, _, n, mr_type]) for _ in kRange],
           methods, title, "k", "Computation Time (sec.)", "t_" + str(n) + "_" + mr_type)

      # COMPARING WITH ALG1 for now
      print 'ratio of finding mmr-q'
      plot(kRange, lambda method: [100.0 * sum(mr[method, k, n, mr_type][_] == mr['alg1', k, n, mr_type][_] for _ in range(validTrials)) / validTrials for k in kRange], lambda _: [0.0] * len(kRange),
           methods, title, "k", "% of Finding a MMR Query", "ratiok_" + str(n) + "_" + mr_type)

  print 'debug'
  validSeq = [_ for _ in range(trials) if _ not in excluded]
  for n in nRange:
    for k in kRange:
      print k, n, {validSeq[r]: mr['alg1', k, n, 'mrk'][r] - mr['chain', k, n, 'mrk'][r] for r in range(validTrials) if
                   mr['alg1', k, n, 'mrk'][r] - mr['chain', k, n, 'mrk'][r] != 0}


def maximumRegretCVSRelPhi():
  mr = {}
  time = {}

  relPhiNum = {}
  for n in nRange:
    for r in range(trials):
      if r in excluded: continue
      domainFileName = 'domain_' + str(n) + '_' + str(r) + '.pkl'
      (relFeats, domPis, domPiTime) = pickle.load(open(domainFileName, 'rb'))
      relPhiNum[n, r] = len(relFeats)
  
  #print relPhiNum

  # granularity of x axis
  gran = 3
  bins = range(max(nRange) / gran + 1)

  for k in kRange:
    print k
    for mr_type in ['mrk']:
      mr_label = '$MR$' if mr_type == 'mr' else '$MR_k$'
      title = "Objective is " + mr_label + ", $k = " + str(k) + "$"
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
            ret = pickle.load(open(method + '_' + mr_type + '_' + str(k) + '_' + str(n) + '_' + str(r) + '.pkl', 'rb'))
            mr[method, k, relPhiNum[n, r] / gran].append(ret[mr_type])
            time[method, k, relPhiNum[n, r] / gran].append(ret['time'])
            
            xScatter[method].append(relPhiNum[n, r])
            yScatter[method].append(ret[mr_type])

      print 'measured by mr/mrk'
      plot([_ * gran for _ in bins], lambda method: [mean(mr[method, k, _]) for _ in bins], lambda method: [standardErr(mr[method, k, _]) for _ in bins],
           methods, title, "|$\Phi_{rel}$|", "Maximum Regret (" + mr_label + ")", "mrc_" + str(k) + "_" + mr_type)

      #scatter(xScatter, yScatter, methods, title, "|$\Phi_{rel}$|", "Maximum Regret (" + mr_label + ")", "mrc_" + str(k) + "_" + mr_type)

      print 'time'
      plot([_ * gran for _ in bins], lambda method: [mean(time[method, k, _]) for _ in bins], lambda method: [standardErr(time[method, k, _]) for _ in bins],
           methods, title, "|$\Phi_{rel}$|", "Computation Time (sec.)", "tc_" + str(k) + "_" + mr_type)

      print 'ratio of finding mmr-q'
      plot([_ * gran for _ in bins], lambda method: [100.0 * sum(mr[method, k, bin][_] == mr['alg1', k, bin][_] for _ in range(len(mr['alg1', k, bin]))) / (len(mr['alg1', k, bin])) if len(mr['alg1', k, bin]) > 0 else nan for bin in bins], lambda _: [0.0] * len(bins),
           methods, title, "|$\Phi_{rel}$|", "% of Finding a MMR Query", "ratioc_" + str(k) + "_" + mr_type)


def regret():
  mr = {}
  
  #methods = ['alg1', 'chain', 'relevantRandom', 'random', 'nq']
  methods = ['alg1']
  legends = ['alg1mr', 'alg1mrk']

  pRange = [0.1, 0.15, 0.2, 0.5, 0.8, 1]

  for n in nRange:
    print n
    for k in kRange:
      print k
      title = "$|\Phi_?| = "+ str(n) + "$, $k = " + str(k) + "$"

      for mr_type in ['mr', 'mrk']:
        for method in methods:
          ret = {}
          for p in pRange:
            mr[method + mr_type, k, n, p] = []

          for r in range(trials):
            if r in excluded: continue
            ret = pickle.load(open(method + '_' + mr_type + '_' + str(k) + '_' + str(n) + '_' + str(r) + '.pkl', 'rb'))
            for p in pRange:
              #if method == 'random' and p == 0.1: print [(_, ret[_]['mr']) for _ in range(trials)]
              mr[method + mr_type, k, n, p].append(ret['regrets'][p])

      plot(pRange, lambda method: [mean(mr[method, k, n, _]) for _ in pRange], lambda method: [standardErr(mr[method, k, n, _]) for _ in pRange],
           legends, title, "% of Changeable Features", "Regret", str(n) + "_" + str(k) + "_regret")

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
    lines = ax.errorbar(x, y(method), yci(method), fmt=markers[method], label=legends[method], mfc='none', markersize=15, capsize=5)

  #ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

  pylab.title(title)
  pylab.xlabel(xlabel)
  pylab.ylabel(ylabel)
  #pylab.ylim([0, 0.2])
  pylab.gcf().subplots_adjust(bottom=0.15, left=0.2)
  fig.savefig(filename + ".pdf", dpi=300, format="pdf")
  
  figLegend = pylab.figure(figsize = (4.5, 3))
  pylab.figlegend(*ax.get_legend_handles_labels(), loc = 'upper left')
  figLegend.savefig("legend.pdf", dpi=300, format="pdf")

def scatter(x, y, methods, title, xlabel, ylabel, filename):
  fig = pylab.figure()

  ax = pylab.gca()
  for method in methods:
    ax.scatter(x[method], y[method], marker=markerStyle[method], color=colorMap[method])

  pylab.title(title)
  pylab.xlabel(xlabel)
  pylab.ylabel(ylabel)
  pylab.gcf().subplots_adjust(bottom=0.15, left=0.2)
  fig.savefig(filename + "_scatter.pdf", dpi=300, format="pdf")
 

def standardErr(data):
  return std(data) / sqrt(len(data))

if __name__ == '__main__':
  font = {'size': 20}
  matplotlib.rc('font', **font)
  
  excludeFailedExperiments()
  
  maximumRegretK()

  #maximumRegretCVSRelPhi()

  #regret()
