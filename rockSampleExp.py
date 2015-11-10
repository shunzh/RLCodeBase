from QTPAgent import AlternatingQTPAgent, JointQTPAgent
from CMPExp import Experiment
import util
import sys
from rockSample import RockSample
import random
import getopt
import config
from cmp import QueryType

"""
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
  horizon = 30
  objNum = 5
  rewardCandNum = 3
  objNumPerFeat = 2
  # discount factor
  gamma = 0.9
  stepCost = 0
  agentName = 'JQTP'
  
  try:
    opts, args = getopt.getopt(sys.argv[1:], "r:l:s:d:a:n:c:")
  except getopt.GetoptError:
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-r':
      random.seed(int(arg))
    elif opt == '-l':
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
    rewardSignal = -1 + 2 * random.random()
    for idx in xrange(objNumPerFeat):
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
  terminalReward = util.Counter()
  terminalReward[(width / 2, 0)] = 100

  def relevance(fState, query):
    return abs(fState[0] - query[0]) + abs(fState[1] - query[1]) < horizon - responseTime

  rewardSet = [rewardGen(reward) for reward in rewards]
  initialPhi = [1.0 / rewardCandNum] * rewardCandNum
  # ask whether reward is good or not
  queryType = QueryType.REWARD_SIGN

  # the true reward function is chosen randomly
  cmp = RockSample(queries, random.choice(rewardSet), gamma, responseTime, width, height,\
                   horizon = horizon, terminalReward = terminalReward)
  if agentName == 'JQTP':
    agent = JointQTPAgent(cmp, rewardSet, initialPhi, queryType, gamma)
  elif agentName == 'AQTP':
    agent = AlternatingQTPAgent(cmp, rewardSet, initialPhi, queryType, relevance, gamma)
  elif agentName == 'AQTP-NF':
    # don't filter query. Assume all queries are relevant.
    agent = AlternatingQTPAgent(cmp, rewardSet, initialPhi, queryType, lambda fS, q: True, gamma)
  elif agentName == 'AQTP-RS':
    agent = AlternatingQTPAgent(cmp, rewardSet, initialPhi, queryType, relevance, gamma, restarts=1)
  else:
    raise Exception("Unknown Agent " + agentName)

  ret, qValue, time = Experiment(cmp, agent, gamma, rewardSet, queryType, horizon=horizon)
  print ret
  print qValue
  print time

if __name__ == '__main__':
  main()
