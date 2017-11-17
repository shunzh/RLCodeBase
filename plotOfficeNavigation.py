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
excluded = [39]

def maximumRegretK(mrk=False):
  mr = {}
  time = {}
  q = {}
  
  if mrk: # no reallyBrute for mrk
    #methods = ['brute', 'alg1', 'chain', 'relevantRandom', 'random', 'nq']
    methods = ['alg1', 'chain', 'relevantRandom', 'random', 'nq']
  else:
    #methods = ['brute', 'alg1', 'chain', 'relevantRandom', 'random', 'nq']
    methods = ['alg1', 'chain', 'relevantRandom', 'random', 'nq']
  
  mr_type = 'mrk' if mrk else 'mr'
  mr_label = " ($MR_k$)" if mrk else ""

  n = 15
  kRange = [0, 1, 2, 3]

  for k in kRange:
    for method in methods:
      mr[method, k, n] = []
      time[method, k, n] = []
      q[method, k, n] = []
      for r in range(trials):
        if r not in excluded:
          ret = pickle.load(open(method + '_' + mr_type + '_' + str(k) + '_' + str(n) + '_' + str(r) + '.pkl', 'rb'))
          mr[method, k, n].append(ret[mr_type])
          time[method, k, n].append(ret['time'])
          q[method, k, n].append(ret['q'])

  plot(kRange, lambda method: [mean(mr[method, _, n]) for _ in kRange], lambda method: [standardErr(mr[method, _, n]) for _ in kRange],
       methods, "k", "Maximum Regret" + mr_label, "mrk_" + mr_type)

  plot(kRange, lambda method: [mean(time[method, _, n]) for _ in kRange], lambda method: [standardErr(time[method, _, n]) for _ in kRange],
       methods, "k", "Computation Time (sec.)", "tk_" + mr_type)

  # COMPARING WITH ALG1 for now1
  validTrials = trials - len(excluded)
  plot(kRange, lambda method: [100.0 * sum(mr[method, k, n][_] == mr['alg1', k, n][_] for _ in range(validTrials)) / validTrials for k in kRange], lambda _: [0.0] * len(kRange),
       methods, "k", "% of Finding a MMR Query", "ratiok")

def maximumRegretC(mrk=False):
  """
  DEPRECATED
  """
  mr = {}
  time = {}

  methods = ['brute', 'alg1', 'chain', 'relevantRandom', 'random', 'nq']
  
  mr_type = " ($MR_k$)" if mrk else ""
  prefix = 'mr_k/' if mrk else ""

  nRange = [5, 10, 15]
  kRange = [2]

  for n in nRange:
    for k in kRange:
      for method in methods:
        mr[method, k, n] = []
        time[method, k, n] = []
        for r in range(trials):
          if r in [36, 80]: continue
          ret = pickle.load(open(prefix + method + '_' + str(k) + '_' + str(n) + '_' + str(r) + '.pkl', 'rb'))
          mr[method, k, n].append(ret['mr'])
          time[method, k, n].append(ret['time'])
  
  validTrials = trials
  plot(nRange, lambda method: [mean(mr[method, 2, _]) for _ in nRange], lambda method: [standardErr(mr[method, 2, _]) for _ in nRange],
       methods, "|$\Phi_?$|", "Maximum Regret" + mr_type, "mrc")

  plot(nRange, lambda method: [mean(time[method, 2, _]) for _ in nRange], lambda method: [standardErr(time[method, 2, _]) for _ in nRange],
       methods, "|$\Phi_?$|", "Computation Time (sec.)", "tc")

  """
  plot(nRange, lambda method: [1.0 * sum(mr[n, 2, method][_] == mr[n, 2, 'brute'][_] for _ in range(validTrials)) / validTrials for n in nRange], lambda _: [0.0] * len(nRange),
       methods, "|$\Phi_?$|", "Ratio of Finding MMR Query", "ratioc")
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

  n = 15
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
       methods, "% of Changeable Features", "Regret", "rp" + mr_type)

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

  maximumRegretK()
  #maximumRegretC()
  #maximumRegretCVSRelPhi()

  maximumRegretK(mrk=True)
  #maximumRegretC(mrk=True)
  #maximumRegretCVSRelPhi(mrk=True)

  regret()
  regret(mrk=True)
