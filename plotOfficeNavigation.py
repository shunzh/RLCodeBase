import pickle
from numpy import mean, std, sqrt

def main():
  mrs = pickle.load(open('mrs', 'rb'))
  times = pickle.load(open('times', 'rb'))
  
  for method in ['brute', 'alg1', 'chain', 'random', 'nq']:
    m, ci = process([mrs[method, _] for _ in range(10)])
    print method, m, ci
    m, ci = process([times[method, _] for _ in range(10)])
    print method, m, ci

def process(data):
  return mean(data), 1.95 * std(data) / sqrt(len(data))

if __name__ == '__main__':
  main()