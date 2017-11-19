import pickle
from numpy import mean, std, sqrt
import matplotlib.pyplot as plt
import matplotlib
import pylab
from matplotlib.ticker import FormatStrFormatter

markers = {'reallyBrute': 'bo--', 'brute': 'bo-',\
           'alg1': 'gs-', 'alg1NoScope': 'gs--', 'alg1NoFilter': 'gs-.',\
           'chain': 'rd-', 'relevantRandom': 'm^-', 'random': 'm^--', 'nq': 'c+-'}

legends = {'reallyBrute': 'Brute Force', 'brute': 'Brute Force (rel. feat.)',\
           'alg1': 'MMRQ-k',  'alg1NoScope': 'Alg. 3 w/ only Thm. 4.2', 'alg1NoFilter': 'Alg. 3 w/ only Thm. 4.1',\
           'chain': 'CoA', 'relevantRandom': 'Random (rel. feat.)', 'random': 'Random', 'nq': 'No Query'}


# shared for all functions
trials = 100
excluded = [13, 39, 71]

def maximumRegretK():
  mr = {}
  mrk = {}
  time = {}
  q = {}
  
  #methods = ['brute', 'alg1', 'chain', 'relevantRandom', 'random', 'nq']
  methods = ['alg1', 'chain', 'relevantRandom', 'random', 'nq']

  n = 20
  kRange = [0, 1, 2, 3]

  for mr_type in ['mr', 'mrk']:
    for k in kRange:
      for method in methods:
        # queries that optimize mr_type and measured by mr
        mr[method, k, n, mr_type] = []
        # queries that optimize mr_type and measured by mrk
        mrk[method, k, n, mr_type] = []
        time[method, k, n, mr_type] = []
        q[method, k, n, mr_type] = []
        for r in range(trials):
          if r not in excluded:
            ret = pickle.load(open(method + '_' + mr_type + '_' + str(k) + '_' + str(n) + '_' + str(r) + '.pkl', 'rb'))
            mr[method, k, n, mr_type].append(ret['mr'])
            mrk[method, k, n, mr_type].append(ret['mrk'])
            time[method, k, n, mr_type].append(ret['time'])
            q[method, k, n, mr_type].append(ret['q'])

    plot(kRange, lambda method: [mean(mr[method, _, n, mr_type]) for _ in kRange], lambda method: [standardErr(mr[method, _, n, mr_type]) for _ in kRange],
         methods, "k", "Maximum Regret ($MR$)", mr_type + '_mr')

    plot(kRange, lambda method: [mean(mrk[method, _, n, mr_type]) for _ in kRange], lambda method: [standardErr(mr[method, _, n, mr_type]) for _ in kRange],
         methods, "k", "Maximum Regret ($MR_k$)", mr_type + '_mrk')

    plot(kRange, lambda method: [mean(time[method, _, n, mr_type]) for _ in kRange], lambda method: [standardErr(time[method, _, n, mr_type]) for _ in kRange],
         methods, "k", "Computation Time (sec.)", mr_type + '_t')

    # COMPARING WITH ALG1 for now1
    validTrials = trials - len(excluded)
    plot(kRange, lambda method: [100.0 * sum(mr[method, k, n, mr_type][_] == mr['alg1', k, n, mr_type][_] for _ in range(validTrials)) / validTrials for k in kRange], lambda _: [0.0] * len(kRange),
         methods, "k", "% of Finding a MMR Query", "ratiok_" + mr_type)

  # for more debugging
  """
  for k in kRange:
    print k, {r: mr['alg1', k, n, 'mr'][r] - mr['alg1', k, n, 'mrk'][r] for r in range(validTrials)}
  """

def maximumRegretCVSRelPhi(mrk=False):
  mr = {}
  time = {}

  #methods = ['brute', 'alg1', 'chain', 'relevantRandom', 'random', 'nq']
  methods = ['alg1', 'chain', 'relevantRandom', 'random', 'nq']
  
  mr_type = 'mrk' if mrk else 'mr'
  mr_label = " ($MR_k$)" if mrk else ""

  nRange = [5, 10, 15]
  k = 2

  relPhiNum = {}
  for n in nRange:
    for r in range(trials):
      relPhiNum[n, r] = pickle.load(open('relPhi_' + str(n) + '_' + str(r) + '.pkl', 'rb'))
      
  validRange = range(trials)
  for n in nRange:
    print "%.2f" % mean([relPhiNum[n, _] for _ in validRange])
    print "%.2f" % standardErr([relPhiNum[n, _] for _ in validRange])

  gran = 4
  bins = range(max(nRange) / gran + 1)
  print bins

  for n in nRange:
    for method in methods:
      for bin in bins: 
        mr[method, k, bin] = []
        time[method, k, bin] = []

      for r in range(trials):
        if r in excluded: continue
        ret = pickle.load(open(method + '_' + mr_type + '_' + str(k) + '_' + str(n) + '_' + str(r) + '.pkl', 'rb'))
        mr[method, k, relPhiNum[n, r] / gran].append(ret['mr'])
        time[method, k, relPhiNum[n, r] / gran].append(ret['time'])

  plot([_ * gran for _ in bins], lambda method: [mean(mr[method, k, _]) for _ in bins], lambda method: [standardErr(mr[method, k, _]) for _ in bins],
       methods, "|$\Phi_{rel}$|", "Maximum Regret" + mr_label, "mrc_phir_" + mr_type)

  plot([_ * gran for _ in bins], lambda method: [mean(time[method, k, _]) for _ in bins], lambda method: [standardErr(time[method, k, _]) for _ in bins],
       methods, "|$\Phi_{rel}$|", "Computation Time (sec.)", "tc_phir_" + mr_type)

  plot([_ * gran for _ in bins], lambda method: [100.0 * sum(mr[method, k, bin][_] == mr['brute', 2, bin][_] for _ in range(len(mr['brute', k, bin]))) / len(mr['brute', 2, bin]) for bin in bins], lambda _: [0.0] * len(bins),
       methods, "|$\Phi_{rel}$|", "% of Finding a MMR Query", "ratioc_phir_" + mr_type)


def regret(mrk=False):
  mr = {}
  
  methods = ['alg1', 'chain', 'relevantRandom', 'random', 'nq']

  mr_type = 'mrk' if mrk else 'mr'

  n = 20
  k = 2
  pRange = [0.1, 0.5, 0.9]

  for method in methods:
    ret = {}
    for p in pRange:
      mr[method, k, n, p] = []
    for r in range(trials):
      if r not in excluded:
        ret = pickle.load(open(method + '_' + mr_type + '_' + str(k) + '_' + str(n) + '_' + str(r) + '.pkl', 'rb'))
        for p in pRange:
          #if method == 'random' and p == 0.1: print [(_, ret[_]['mr']) for _ in range(trials)]
          mr[method, k, n, p].append(ret['regrets'][p])

  plot(pRange, lambda method: [mean(mr[method, k, n, _]) for _ in pRange], lambda method: [standardErr(mr[method, k, n, _]) for _ in pRange],
       methods, "% of Changeable Features", "Regret", mr_type + "_regret")

def printTex(y, yci, t, tci, methods, legends):
  for i in range(len(methods)):
    method = methods[i]
    print legends[i], '& $', round(y(method), 4), '\pm', round(yci(method), 4), '$ &',
    print '$', round(t(method), 4), '\pm', round(tci(method), 4), '$ \\\\'

def plot(x, y, yci, methods, xlabel, ylabel, filename):
  fig = pylab.figure()

  ax = pylab.gca()
  for method in methods:
    print method, y(method), yci(method)
    lines = ax.errorbar(x, y(method), yci(method), fmt=markers[method], label=legends[method], mfc='none', markersize=15, capsize=5)
    #lines = ax.plot(x, y(method), marker=markers[method], color=colors[method], linestyle=linestyles[method], mfc='none', label=legends[method], markersize=15)

  #ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

  #plt.legend(legends)
  pylab.xlabel(xlabel)
  pylab.ylabel(ylabel)
  #pylab.ylim([-.2, 4])
  pylab.gcf().subplots_adjust(bottom=0.15, left=0.2)
  fig.savefig(filename + ".pdf", dpi=300, format="pdf")
  
  figLegend = pylab.figure(figsize = (4.5, 3))
  pylab.figlegend(*ax.get_legend_handles_labels(), loc = 'upper left')
  figLegend.savefig("legend.pdf", dpi=300, format="pdf")


def standardErr(data):
  return std(data) / sqrt(len(data))

if __name__ == '__main__':
  font = {'size': 20}
  matplotlib.rc('font', **font)

  #maximumRegretK()

  #maximumRegretCVSRelPhi()
  #maximumRegretCVSRelPhi(mrk=True)

  regret()
  regret(mrk=True)
