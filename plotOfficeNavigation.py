import pickle
from numpy import mean, std, sqrt
import matplotlib.pyplot as plt
import matplotlib
import pylab
from matplotlib.ticker import FormatStrFormatter

markers = {'reallyBrute': 'bo--', 'brute': 'bo-',\
           'alg1': 'gs-', 'alg1NoScope': 'gs--', 'alg1NoFilter': 'gs-.',\
           'chain': 'rd-', 'random': 'm^-', 'nq': 'c+-'}

legends = {'reallyBrute': 'Brute Force', 'brute': 'Brute Force (rel. cons.)',\
           'alg1': 'Alg. 3',  'alg1NoScope': 'Alg. 3 w/ only Thm. 4.2', 'alg1NoFilter': 'Alg. 3 w/ only Thm. 4.1',\
           'chain': 'CoA', 'random': 'Random', 'nq': 'No Query'}

def maximumRegretK(mrk=False):
  trials = 20
  m = {}
  ci = {}
  tm = {}
  tci = {}

  mr = {}
  time = {}
  
  if mrk: # no reallyBrute for mrk
    methods = ['brute', 'alg1', 'alg1NoFilter', 'alg1NoScope', 'chain', 'random', 'nq']
    #methods = ['brute', 'alg1', 'chain', 'random', 'nq']
  else:
    methods = ['reallyBrute', 'brute', 'alg1', 'alg1NoFilter', 'alg1NoScope', 'chain', 'random', 'nq']
  
  mr_type = " ($MR_k$)" if mrk else ""

  nRange = [10]
  kRange = [0, 1, 2, 3]

  for n in nRange:
    for k in kRange:
      for method in methods:
        mr[n, k, method] = []
        time[n, k, method] = []
        for r in range(trials):
          ret = pickle.load(open(method + '_' + str(k) + '_' + str(n) + '_' + str(r) + '.pkl', 'rb'))
          mr[n, k, method].append(ret['mr'])
          time[n, k, method].append(ret['time'])

        m[method, k, n], ci[method, k, n] = process(mr[n, k, method])
        tm[method, k, n], tci[method, k, n] = process(time[n, k, method])

  plot(kRange, lambda method: [m[method, _, 10] for _ in kRange], lambda method: [ci[method, _, 10] for _ in kRange],
       methods, "k", "Maximum Regret" + mr_type, "mrk")

  plot(kRange, lambda method: [tm[method, _, 10] for _ in kRange], lambda method: [tci[method, _, 10] for _ in kRange],
       methods, "k", "Computation Time (sec.)", "tk")

  plot(kRange, lambda method: [1.0 * sum(mr[10, k, method][_] == mr[10, k, 'brute'][_] for _ in range(trials)) / trials for k in kRange], lambda _: [],
       methods, "k", "Ratio of Finding MMR Query", "ratiok")

def maximumRegretC(mrk=False):
  trials = 20
  m = {}
  ci = {}
  tm = {}
  tci = {}
  
  mr = {}
  time = {}

  methods = ['brute', 'alg1', 'alg1NoFilter', 'alg1NoScope', 'chain', 'random', 'nq']
  
  mr_type = " ($MR_k$)" if mrk else ""

  nRange = [5, 10, 15]
  kRange = [2]

  for n in nRange:
    for k in kRange:
      for method in methods:
        mr[n, k, method] = []
        time[n, k, method] = []
        for r in range(trials):
          ret = pickle.load(open(method + '_' + str(k) + '_' + str(n) + '_' + str(r) + '.pkl', 'rb'))
          mr[n, k, method].append(ret['mr'])
          time[n, k, method].append(ret['time'])

        m[method, k, n], ci[method, k, n] = process(mr[n, k, method])
        tm[method, k, n], tci[method, k, n] = process(time[n, k, method])

  plot(nRange, lambda method: [m[method, 2, _] for _ in nRange], lambda method: [ci[method, 2, _] for _ in nRange],
       methods, "|$\Phi_?$|", "Maximum Regret" + mr_type, "mrc")

  plot(nRange, lambda method: [tm[method, 2, _] for _ in nRange], lambda method: [tci[method, 2, _] for _ in nRange],
       methods, "|$\Phi_?$|", "Computation Time (sec.)", "tc")

  plot(nRange, lambda method: [1.0 * sum(mr[n, 2, method][_] == mr[n, 2, 'brute'][_] for _ in range(trials)) / trials for n in nRange], lambda _: [],
       methods, "|$\Phi_?$|", "Ratio of Finding MMR Query", "ratioc")

def regret():
  trials = 20
  m = {}
  ci = {}
  tm = {}
  tci = {}
  
  methods = ['alg1', 'chain', 'random', 'nq']

  n = 10
  k = 2
  pRange = [0.1, 0.5, 0.9]

  for method in methods:
    for p in pRange:
      ret = {}
      for r in range(trials):
        ret[r] = pickle.load(open(method + '_' + str(k) + '_' + str(n) + '_' + str(p) + '_' + str(r) + '.pkl', 'rb'))
      m[method, k, n, p], ci[method, k, n, p] = process([ret[_]['mr'] for _ in range(trials)])
      tm[method, k, n, p], tci[method, k, n, p] = process([ret[_]['time'] for _ in range(trials)])

  plot(pRange, lambda method: [m[method, k, n, _] for _ in pRange], lambda method: [ci[method, k, n, _] for _ in pRange],
       methods, "Ratio of Changeable Features", "Regret", "rp")

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

  ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

  #plt.legend(legends)
  pylab.xlabel(xlabel)
  pylab.ylabel(ylabel)
  #pylab.ylim([-.2, 4])
  pylab.gcf().subplots_adjust(bottom=0.15, left=0.2)
  fig.savefig(filename + ".pdf", dpi=300, format="pdf")
  
  figLegend = pylab.figure(figsize = (4.5, 3.5))
  pylab.figlegend(*ax.get_legend_handles_labels(), loc = 'upper left')
  figLegend.savefig("legend.pdf", dpi=300, format="pdf")


def process(data):
  return mean(data), std(data) / sqrt(len(data))

if __name__ == '__main__':
  font = {'size': 20}
  matplotlib.rc('font', **font)

  maximumRegretK()
  #maximumRegretC()

  #maximumRegretK(mrk=True)
  #maximumRegretC(mrk=True)

  #regret()
