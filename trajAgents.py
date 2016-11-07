from QTPAgent import MILPAgent, QTPAgent
import easyDomains
import config
import util
import random
import copy
import scipy.stats
import numpy

# helper for MILP agent
class MILPTrajAgent(MILPAgent):
  def learn(self):
    args, q = MILPAgent.learn(self)
    rewardCandNum = len(self.rewardSet)
    k = config.NUMBER_OF_RESPONSES

    hValues = util.Counter()
    hTrajs = util.Counter()
    policyBins = MILPAgent.computeDominatingPis(args, q)

    for s in args['S']:
      us = []
      for i in xrange(rewardCandNum):
        us.append(self.sampleTrajectory(self.viAgentSet[i].x, s, hori=config.TRAJECTORY_LENGTH, to='trajectory'))
      # avoid comparing trajectories of different lengths
      if any(len(u) < config.TRAJECTORY_LENGTH for u in us):
        continue
        
      hTraj = []
      if config.GENERATE_RANDOM_TRAJ:
        # WAY 1: generating random policies
        hTraj = [tuple(self.sampleTrajectory(None, s, hori=config.TRAJECTORY_LENGTH, to='trajectory')) for _ in xrange(k)]
      else:
        # WAY 2: using prefixes of optimal policies of some reward candidates
        available = [True] * rewardCandNum
        for i in xrange(k):
          subPsi = copy.copy(self.phi)
          subPsi = [subPsi[idx] if policyBins[idx][i] > 0 and available[idx] else 0 for idx in xrange(rewardCandNum)]
          if sum(subPsi) == 0:
            # this means the agent does not need more trajs in the query
            # then just add a random one to make this a k-nary response
            hTraj.append(tuple(self.sampleTrajectory(None, s, hori=config.TRAJECTORY_LENGTH, to='trajectory')))
          else:
            idx = util.sample(subPsi, range(rewardCandNum))
            # it's possible that one policy dominates multiple reward candidates..
            # to avoid containing same policies in the query as a result of this
            # mark any policy added to the traj query as unavailable
            available[idx] = False
            if config.VERBOSE: print s, i, subPsi, idx
            hTraj.append(tuple(us[idx]))

      psiProbs = self.getPossiblePhiAndProbs(hTraj)
      hValue = 0
      for psi, prob in psiProbs:
        bins = [0] * k
        for idx in xrange(rewardCandNum):
          if psi[idx] > 0:
            # put opt policies into bins
            bins = [sum(_) for _ in zip(bins, policyBins[idx])]
        hValue += prob * scipy.stats.entropy(bins)
      if config.VERBOSE: print hValue, psiProbs
      
      if s in hValues.keys():
        if hValue < hValues[s]: continue
      
      hValues[s] = hValue
      hTrajs[s] = hTraj
    
    minH = min(hValues.values())
    minStates = filter(lambda _: hValues[_] == minH, hValues.keys())
    q = hTrajs[random.choice(minStates)]
    objValue = self.getQValue(self.cmp.state, None, q)
    return (q, None, objValue)


class DisagreeTrajAgent(QTPAgent):
  """
  Compute the distance between optimal trajectories.
  
  from Wilson et al.
  """
  def learn(self):
    args = easyDomains.convert(self.cmp, self.rewardSet, self.phi)
    rewardCandNum = len(self.rewardSet)
    k = config.NUMBER_OF_RESPONSES

    hValues = util.Counter()
    hTrajs = util.Counter()

    for s in args['S']:
      us = []
      for i in xrange(rewardCandNum):
        us.append(self.sampleTrajectory(self.viAgentSet[i].x, s, hori=config.TRAJECTORY_LENGTH, to='trajectory'))
      # avoid comparing trajectories of different lengths
      if any(len(u) < config.TRAJECTORY_LENGTH for u in us):
        continue

      if config.GENERATE_RANDOM_TRAJ:
        hTrajs[s] = [tuple(self.sampleTrajectory(None, s, hori=config.TRAJECTORY_LENGTH, to='trajectory')) for _ in xrange(k)]
      else:
        indices = numpy.random.choice(range(rewardCandNum), k, replace=False)
        hTrajs[s] = [tuple(us[idx]) for idx in indices]
     
      for i in xrange(rewardCandNum):
        for j in xrange(rewardCandNum):
          hValues[s] += self.cmp.getTrajectoryDistance(us[i], us[j])
    maxH = max(hValues.values())
    maxStates = filter(lambda _: hValues[_] == maxH, hValues.keys())
    s = random.choice(maxStates)

    objValue = self.getQValue(self.cmp.state, None, hTrajs[s])
    return (hTrajs[s], None, objValue)


class BeliefChangeTrajAgent(QTPAgent):
  """
  We compute the posterior belief, but only compare it with the prior belief,
  not using the desirable belief.

  from Wilson et al.
  """
  def learn(self):
    args = easyDomains.convert(self.cmp, self.rewardSet, self.phi)
    rewardCandNum = len(self.rewardSet)
    k = config.NUMBER_OF_RESPONSES

    hValues = util.Counter()
    hTrajs = util.Counter()

    for s in args['S']:
      us = []
      for i in xrange(rewardCandNum):
        us.append(self.sampleTrajectory(self.viAgentSet[i].x, s, hori=config.TRAJECTORY_LENGTH, to='trajectory'))
      # avoid comparing trajectories of different lengths
      if any(len(u) < config.TRAJECTORY_LENGTH for u in us):
        continue
      
      if config.GENERATE_RANDOM_TRAJ:
        hTrajs[s] = [tuple(self.sampleTrajectory(None, s, hori=config.TRAJECTORY_LENGTH, to='trajectory')) for _ in xrange(k)]
      else:
        indices = numpy.random.choice(range(rewardCandNum), k, replace=False)
        hTrajs[s] = [tuple(us[idx]) for idx in indices]
      
      # compute the different between new psi and old psi
      psiProbs = self.getPossiblePhiAndProbs(hTrajs[s])
      for psi, prob in psiProbs:
        hValues[s] += prob * sum(abs(p1 - p2) for p1, p2 in zip(psi, self.phi))

    maxH = max(hValues.values())
    maxStates = filter(lambda _: hValues[_] == maxH, hValues.keys())
    q = hTrajs[random.choice(maxStates)]
    objValue = self.getQValue(self.cmp.state, None, q)
    return (q, None, objValue)


class RandomTrajAgent(QTPAgent):
  def learn(self):
    args = easyDomains.convert(self.cmp, self.rewardSet, self.phi)
    k = config.NUMBER_OF_RESPONSES

    while True:
      s = random.choice(args['S'])
      q = [tuple(self.sampleTrajectory(None, s, hori=config.TRAJECTORY_LENGTH, to='trajectory')) for _ in xrange(k)]
      if any(len(u) < config.TRAJECTORY_LENGTH for u in q): continue
      else: break
    objValue = self.getQValue(self.cmp.state, None, q)
    return (q, None, objValue)
