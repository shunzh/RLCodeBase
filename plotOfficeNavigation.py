import pickle
from numpy import mean, std, sqrt
import matplotlib.pyplot as plt

def main():
  trials = 20
  m = {}
  ci = {}
  
  nRange = [5, 10, 15]
  kRange = [2, 3, 4]
  methods = ['brute', 'alg1', 'chain', 'random', 'nq']

  for n in nRange:
    for k in kRange:
      for method in methods:
        try:
          ret = pickle.load(open(method + '_' + str(k) + '_' + str(n) + '.pkl', 'rb'))
          m[method, k, n], ci[method, k, n] = process([ret['mr', _] for _ in range(trials)])
        except:
          print 'unable to read', n, k, method

  plt.figure()
  for method in methods:
    print method, [m[method, _, 10] for _ in kRange], [ci[method, _, 10] for _ in kRange]
    plt.errorbar(kRange, [m[method, _, 10] for _ in kRange], [ci[method, _, 10] for _ in kRange])
  plt.title("Maximum regret under different k")
  plt.legend(methods)
  plt.xlabel('k')
  plt.ylabel('Maximum Regret')
  plt.show()

  plt.figure()
  plt.errorbar(nRange[:2], [m['brute', 2, _] for _ in nRange[:2]], [ci['brute', 2, _] for _ in nRange[:2]])
  for method in methods[1:]:
    print method, [m[method, 2, _] for _ in nRange], [ci[method, 2, _] for _ in nRange]
    plt.errorbar(nRange, [m[method, 2, _] for _ in nRange], [ci[method, 2, _] for _ in nRange])
  plt.title("Maximum regret under different |C|")
  plt.xlabel('|C|')
  plt.ylabel('Maximum Regret')
  plt.legend(methods)
  plt.show()


def process(data):
  return mean(data), 1.95 * std(data) / sqrt(len(data))

if __name__ == '__main__':
  main()
