import pickle
from numpy import mean, std, sqrt
import matplotlib.pyplot as plt
import matplotlib

markers = {'brute': 'o-', 'alg1': '+-', 'chain': 'd-', 'random': '^-', 'nq': '--'}

def maximumRegret():
  trials = 20
  m = {}
  ci = {}
  tm = {}
  tci = {}
  
  methods = ['brute', 'alg1', 'chain', 'random', 'nq']
  legends = ['Brute Force', 'Alg.1', 'CoA', 'Random', 'No Query']

  nRange = [5, 10]
  kRange = [2, 3, 4]

  for n in nRange:
    for k in kRange:
      for method in methods:
        try:
          ret = pickle.load(open(method + '_' + str(k) + '_' + str(n) + '.pkl', 'rb'))
          m[method, k, n], ci[method, k, n] = process([ret['mr', _] for _ in range(trials)])
          tm[method, k, n], tci[method, k, n] = process([ret['time', _] for _ in range(trials)])
        except:
          print 'unable to read', n, k, method

  plot(kRange, lambda method: [m[method, _, 10] for _ in kRange], lambda method: [ci[method, _, 10] for _ in kRange],
       methods, legends, "k", "Maximum Regret", "mrk")

  plot(kRange, lambda method: [tm[method, _, 10] for _ in kRange], lambda method: [tci[method, _, 10] for _ in kRange],
       methods, legends, "k", "Computation Time (sec.)", "tk")

  plot(nRange, lambda method: [m[method, 2, _] for _ in nRange], lambda method: [ci[method, 2, _] for _ in nRange],
       methods, legends, "|C|", "Maximum Regret", "mrc")

  plot(nRange, lambda method: [tm[method, 2, _] for _ in nRange], lambda method: [tci[method, 2, _] for _ in nRange],
       methods, legends, "|C|", "Computation Time (sec.)", "tc")

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
      try:
        ret = pickle.load(open(method + '_' + str(k) + '_' + str(n) + '_' + str(p) + '.pkl', 'rb'))
        m[method, k, n, p], ci[method, k, n, p] = process([ret['mr', _] for _ in range(trials)])
        tm[method, k, n, p], tci[method, k, n, p] = process([ret['time', _] for _ in range(trials)])
      except:
        print 'unable to read', n, k, method

  plot(pRange, lambda method: [m[method, k, n, _] for _ in pRange], lambda method: [ci[method, k, n, _] for _ in pRange],
       methods, legends, "Ratio of violable constraints", "Regret", "rp")


def plot(x, y, yci, methods, legends, xlabel, ylabel, filename):
  plt.figure()
  for method in methods:
    print method, y(method), yci(method)
    plt.errorbar(x, y(method), yci(method), fmt=markers[method], markersize=8)
  plt.legend(legends)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.gcf().subplots_adjust(bottom=0.15, left=0.15)
  plt.savefig(filename + ".pdf", dpi=300, format="pdf")
  plt.close()

def process(data):
  return mean(data), 1.95 * std(data) / sqrt(len(data))

if __name__ == '__main__':
  font = {'size': 12}
  matplotlib.rc('font', **font)

  regret()
  #maximumRegret()
