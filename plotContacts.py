"""
Bar chart demo with pairs of bars grouped for easy comparison.
"""
import numpy as np
import matplotlib.pyplot as plt
from humanWorld import HumanEnvironment

n_groups = 2

def plot(mean_targs, std_targs, mean_obsts, std_obsts, filename):
  """
  Plot a figure with given stats
  """
  index = np.arange(n_groups) * 0.3
  bar_width = 0.1

  error_config = {'ecolor': '0.3'}

  rects1 = plt.bar(index, mean_targs, bar_width,
                   color='#504D9D',
                   yerr=std_targs,
                   error_kw=error_config,
                   label='Targets')

  rects2 = plt.bar(index + bar_width, mean_obsts, bar_width,
                   color='#994D4F',
                   yerr=std_obsts,
                   error_kw=error_config,
                   label='Obstacles')

  plt.ylabel('Number of Objects Contacted')
  plt.xticks(index + bar_width, ('Human', 'Model'))
  plt.axis([0, 0.5, 0, 12])
  plt.legend()
  plt.gcf().set_size_inches(4,4)

  plt.savefig(filename)
  plt.close()

# human data
h_mean_targs = [2.34, 3.03, 10.19, 9.88] 
h_mean_obsts = [2.13, 0.13, 2.28, 0.03]

h_std_targs = [1.49, 1.47, 0.95, 1.33]
h_std_obsts = [1.29, 0.29, 1.20, 0.09]

# model data
#FIXME should read from data file
m_mean_targs = [1.25, 4.13, 6.00, 8.88]
m_mean_obsts = [1.63, 1.38, 2.75, 1.25]

m_std_targs = [1.39, 1.89, 1.51, 1.36]
m_std_obsts = [0.92, 0.92, 0.89, 1.49]

for i in range(4):
  plot((h_mean_targs[i], m_mean_targs[i]), (h_std_targs[i], m_std_targs[i]),
       (h_mean_obsts[i], m_mean_obsts[i]), (h_std_obsts[i], m_std_obsts[i]),
       'contact' + str(i + 1) + '.png')