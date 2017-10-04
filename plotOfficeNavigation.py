import pickle
from numpy import mean, std, sqrt

def main():
  for method in ['brute', 'alg1', 'chain', 'random', 'nq']:
    ret = pickle.load(open(method + '.pkl', 'rb'))
    m, ci = process([ret['mr', _] for _ in range(20)])
    print method, '& $', round(m, 4), '\pm', round(ci, 4), '$ &',
    m, ci = process([ret['time', _] for _ in range(20)])
    print '$', round(m, 4), '\pm', round(ci, 4), '$ \\\\'

def process(data):
  return mean(data), 1.95 * std(data) / sqrt(len(data))

if __name__ == '__main__':
  main()