import pickle
from numpy import mean, std, sqrt
import matplotlib.pyplot as plt
import matplotlib
import pylab

markers = {'reallyBrute': 'bo--', 'brute': 'bo-', 'alg1': 'gs-', 'chain': 'rd-', 'random': 'm^-', 'nq': 'c+-'}
legends = {'reallyBrute': 'Brute Force', 'brute': 'Brute Force (rel. cons.)', 'alg1': 'Proposed Algorithm', 'chain': 'CoA', 'random': 'Random', 'nq': 'No Query'}

def maximumRegret():
  trials = 20
  m = {}
  ci = {}
  tm = {}
  tci = {}
  
  #methods = ['reallyBrute', 'brute', 'alg1', 'chain', 'random', 'nq']
  methods = ['reallyBrute', 'brute', 'alg1', 'chain', 'random', 'nq']
  
  mr_type = ""
  #mr_type = " ($MR_k$)"

  nRange = [10]
  kRange = [2]

  for n in nRange:
    for k in kRange:
      for method in methods:
        if method == 'reallyBrute':
          ret = pickle.load(open(method + '_' + str(k) + '_' + str(n) + '.pkl', 'rb'))
          m[method, k, n], ci[method, k, n] = process([ret['mr', _] for _ in range(trials)])
          tm[method, k, n], tci[method, k, n] = process([ret['time', _] for _ in range(trials)])
        else:
          ret = {}
          for r in range(trials):
            ret[r] = pickle.load(open(method + '_' + str(k) + '_' + str(n) + '_' + str(r) + '.pkl', 'rb'))
          m[method, k, n], ci[method, k, n] = process([ret[_]['mr'] for _ in range(trials)])
          tm[method, k, n], tci[method, k, n] = process([ret[_]['time'] for _ in range(trials)])

  plot(kRange, lambda method: [m[method, _, 10] for _ in kRange], lambda method: [ci[method, _, 10] for _ in kRange],
       methods, "k", "Maximum Regret" + mr_type, "mrk")

  plot(kRange, lambda method: [tm[method, _, 10] for _ in kRange], lambda method: [tci[method, _, 10] for _ in kRange],
       methods, "k", "Computation Time (sec.)", "tk")
  """

  plot(nRange, lambda method: [m[method, 2, _] for _ in nRange], lambda method: [ci[method, 2, _] for _ in nRange],
       methods, "|$\Phi_?$|", "Maximum Regret" + mr_type, "mrc")

  plot(nRange, lambda method: [tm[method, 2, _] for _ in nRange], lambda method: [tci[method, 2, _] for _ in nRange],
       methods, "|$\Phi_?$|", "Computation Time (sec.)", "tc")

  plot(nRange, lambda method: [m[method, 1, _] for _ in nRange], lambda method: [ci[method, 1, _] for _ in nRange],
       methods, "|$\Phi$|", "Maximum Regret", "mrc_k1")

  plot(nRange, lambda method: [tm[method, 1, _] for _ in nRange], lambda method: [tci[method, 1, _] for _ in nRange],
       methods, "|$\Phi$|", "Computation Time (sec.)", "tc_k1")
  """

def regret():
  trials = 10
  m = {}
  ci = {}
  tm = {}
  tci = {}
  
  methods = ['alg1', 'chain', 'random', 'nq']
  legends = ['Alg.1', 'CoA', 'Random', 'No Query']

  n = 10
  k = 1
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
  
  figLegend = pylab.figure(figsize = (4.5, 3))
  pylab.figlegend(*ax.get_legend_handles_labels(), loc = 'upper left')
  figLegend.savefig("legend.pdf", dpi=300, format="pdf")


def process(data):
  return mean(data), 1.95 * std(data) / sqrt(len(data))

if __name__ == '__main__':
  font = {'size': 20}
  matplotlib.rc('font', **font)

  #regret()
  maximumRegret()
