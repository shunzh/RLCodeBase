from QTPAgent import AlternatingQTPAgent, JointQTPAgent
from CMPExp import Experiment
import util
import sys
from rockSample import RockSample
import random
import getopt
import config

"""
Plan:
Algorithms:
- E-JQTP
- AQTP
- AQTP-NF
- AQTP-RS (?)
- opt query
- random query
- no query

Setting:
- response time (axis)

Show:
- expectation Q
- computation time
- paired difference between JQTP and AQTP
"""
def main():
  width = 10
  height = 10
  # the time step that the agent receives the response
  responseTime = 10
  horizon = 20
  objNum = 5
  rewardCandNum = 3
  # discount factor
  gamma = 0.9
  stepCost = 0
  agentName = 'JQTP'
  
  try:
    opts, args = getopt.getopt(sys.argv[1:], "l:s:d:a:n:c:")
  except getopt.GetoptError:
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-l':
      responseTime = int(arg)
    elif opt == '-s':
      width = height = int(arg)
      horizon = width + height
    elif opt == '-d':
      gamma = float(arg)
    elif opt == '-a':
      agentName = arg
    elif opt == '-n':
      objNum = int(arg)
    elif opt == '-c':
      stepCost = float(arg)
  
  # sanity check
  assert horizon > responseTime

  queries = []
  for _ in xrange(objNum):
    x = int(width * random.random())
    y = int(height * random.random())
    queries.append((x, y))
  rewards = []
  for _ in xrange(rewardCandNum):
    reward = util.Counter()
    rewardSignal = random.random()
    for idx in xrange(2):
      q = random.choice(queries)
      reward[q] = rewardSignal 
    rewards.append(reward)
    
  def rewardGen(rewards): 
    def rewardFunc(s):
      if s in rewards.keys():
        return rewards[s]
      else:
        return stepCost
    return rewardFunc

  def relevance(fState, query):
    return abs(fState[0] - query[0]) + abs(fState[1] - query[1]) < horizon - responseTime

  rewardSet = [rewardGen(reward) for reward in rewards]
  initialPhi = [1.0 / rewardCandNum] * rewardCandNum

  cmp = RockSample(queries, rewardSet[0], gamma, responseTime, width, height)
  if agentName == 'JQTP':
    agent = JointQTPAgent(cmp, rewardSet, initialPhi, gamma=gamma)
  elif agentName == 'AQTP':
    agent = AlternatingQTPAgent(cmp, rewardSet, initialPhi, relevance, gamma=gamma)
  elif agentName == 'AQTP-NF':
    agent = AlternatingQTPAgent(cmp, rewardSet, initialPhi, relevance, gamma=gamma, filterQuery=False)
  elif agentName == 'AQTP-RS':
    agent = AlternatingQTPAgent(cmp, rewardSet, initialPhi, relevance, gamma=gamma, restarts=1)
  else:
    raise Exception("Unknown Agent " + agentName)

  ret, qValue, time = Experiment(cmp, agent, gamma, rewardSet, horizon=horizon)
  print ret
  print qValue
  print time

if __name__ == '__main__':
  main()
