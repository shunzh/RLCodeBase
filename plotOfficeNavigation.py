import pickle
from numpy import mean, std, sqrt
import matplotlib.pyplot as plt
import matplotlib
import pylab

markers = {'brute': 'bo-', 'alg1': 'gs-', 'chain': 'rd-', 'random': 'm^-', 'nq': 'c+-'}
legends = {'brute': 'Brute Force', 'alg1': 'Alg.1', 'chain': 'CoA', 'random': 'Random', 'nq': 'No Query'}

def maximumRegret():
  trials = 20
  m = {}
  ci = {}
  tm = {}
  tci = {}
  
  methods = ['brute', 'alg1', 'chain', 'random', 'nq']

  nRange = [5]
  kRange = [0, 1, 2, 3]

  for n in nRange:
    for k in kRange:
      for method in methods:
        ret = {}
        for r in range(trials):
          ret[r] = pickle.load(open(method + '_' + str(k) + '_' + str(n) + '_' + str(r) + '.pkl', 'rb'))
        m[method, k, n], ci[method, k, n] = process([ret[_]['mr'] for _ in range(trials)])
        tm[method, k, n], tci[method, k, n] = process([ret[_]['time'] for _ in range(trials)])

  plot(kRange, lambda method: [m[method, _, 5] for _ in kRange], lambda method: [ci[method, _, 5] for _ in kRange],
       methods, "k", "Maximum Regret", "mrk")

  plot(kRange, lambda method: [tm[method, _, 5] for _ in kRange], lambda method: [tci[method, _, 5] for _ in kRange],
       methods, "k", "Computation Time (sec.)", "tk")

  """
  plot(nRange, lambda method: [m[method, 2, _] for _ in nRange], lambda method: [ci[method, 2, _] for _ in nRange],
       methods, "|C|", "Maximum Regret", "mrc")

  plot(nRange, lambda method: [tm[method, 2, _] for _ in nRange], lambda method: [tci[method, 2, _] for _ in nRange],
       methods, "|C|", "Computation Time (sec.)", "tc")
  """

def regret():
  trials = 20
  m = {}
  ci = {}
  tm = {}
  tci = {}
  
  methods = ['alg1', 'chain', 'random', 'nq']
  legends = ['Alg.1', 'CoA', 'Random', 'No Query']

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
       methods, "Ratio of violable constraints", "Regret", "rp")

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
  #plt.legend(legends)
  pylab.xlabel(xlabel)
  pylab.ylabel(ylabel)
  pylab.gcf().subplots_adjust(bottom=0.15, left=0.15)
  fig.savefig(filename + ".pdf", dpi=300, format="pdf")
  
  figLegend = pylab.figure(figsize = (3, 2.5))
  pylab.figlegend(*ax.get_legend_handles_labels(), loc = 'upper left')
  figLegend.savefig("legend.pdf", dpi=300, format="pdf")


def process(data):
  return mean(data), 1.95 * std(data) / sqrt(len(data))

if __name__ == '__main__':
  font = {'size': 20}
  matplotlib.rc('font', **font)

  regret()
  #maximumRegret()
