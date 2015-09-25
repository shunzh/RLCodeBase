from QTPAgent import IterativeQTPAgent, JointQTPAgent
from windyCorridor import WindyCorridor
from CMPExp import Experiment

# discount factor
gamma = 0.9
# the time step that the agent receives the response
responseTime = 5

interLength = 3
interNum = 3
circular = False

# at this intersection, what would you do?
queries = [(interId, interLength - 1) for interId in xrange(interNum)]

def main():
  rewardSet = [rewardGen([(2, 'L'),(1, 'L'),(0, 'L')]),\
               rewardGen([(2, 'R'),(1, 'R'),(0, 'R')])]

  rewardNum = len(rewardSet)
  initialPhi = [1.0 / rewardNum] * rewardNum

  Agent = JointQTPAgent
  cmp = WindyCorridor(queries, rewardSet[0], gamma, responseTime, interLength, interNum, circular)
  agent = Agent(cmp, rewardSet, initialPhi, gamma=gamma)
 
  ret, qValue = Experiment(cmp, agent, gamma, rewardSet)
  print ret
  print qValue

def rewardGen(pref): 
  def rewardFunc(s):
    if s == pref[0]: return 10
    elif s == pref[1]: return 5
    elif s == pref[2]: return 2
    else: return 0
  
  return rewardFunc

if __name__ == '__main__':
  main()
