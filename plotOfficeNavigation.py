import pickle
from numpy import mean, std, sqrt
import matplotlib.pyplot as plt

methods = ['brute', 'alg1', 'chain', 'random', 'nq']
markers = {'brute': 'o-', 'alg1': '+-', 'chain': 'd-', 'random': '^-', 'nq': '--'}
legends = ['Brute Force', 'Alg.3', 'CoA', 'Random', 'No Query']

def main():
  trials = 20
  m = {}
  ci = {}
  tm = {}
  tci = {}
  
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
       "Maximum regret under different k", "k", "Maximum Regret", "mrk")

  plot(kRange, lambda method: [tm[method, _, 10] for _ in kRange], lambda method: [tci[method, _, 10] for _ in kRange],
       "Computation time under different k", "k", "Computation Time (sec.)", "tk")

  plot(nRange, lambda method: [m[method, 2, _] for _ in nRange], lambda method: [ci[method, 2, _] for _ in nRange],
       "Maximum regret under different |C|", "|C|", "Maximum Regret", "mrc")

  plot(nRange, lambda method: [tm[method, 2, _] for _ in nRange], lambda method: [tci[method, 2, _] for _ in nRange],
       "Computation time under different |C|", "|C|", "Computation Time (sec.)", "tc")

def plot(x, y, yci, title, xlabel, ylabel, filename):
  plt.figure()
  for method in methods:
    print method, y(method), yci(method)
    plt.errorbar(x, y(method), yci(method), fmt=markers[method], markersize=8)
  plt.title(title)
  plt.gcf().set_size_inches(4.5,4)
  plt.legend(legends)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.savefig(filename + ".pdf", dpi=300, format="pdf")
  plt.close()

def process(data):
  return mean(data), 1.95 * std(data) / sqrt(len(data))

if __name__ == '__main__':
  main()
