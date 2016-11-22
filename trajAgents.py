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
    policyBins = MILPAgent.computeDominatingPis(args, q)

    for s in args['S']:
      indices = []
      for i in xrange(k):
        subPsi = [self.phi[idx] if policyBins[idx][i] > 0 and not idx in indices else 0 for idx in xrange(rewardCandNum)]
        if sum(subPsi) == 0:
          # this means the agent does not need more trajs in the query
          # then just add a random opt policy from the available
          random.choice([idx for idx in xrange(rewardCandNum) if not idx in indices])
          indices.append(idx)
        else:
          idx = util.sample(subPsi, range(rewardCandNum))
          # it's possible that one policy dominates multiple reward candidates..
          # to avoid containing same policies in the query as a result of this
          # mark any policy added to the traj query as unavailable
          if config.DEBUG: print s, i, subPsi, idx
          indices.append(idx)

      # now we have a set of reward candidates to sample trajectories..
      # fix them, and generate trajectories for several times 
      indices = tuple(indices)
      for sampleIdx in xrange(config.SAMPLES_TIMES):
        trajs = [self.sampleTrajFromRewardCandidate(idx, s) for idx in indices]
        if any(len(u) < config.TRAJECTORY_LENGTH for u in trajs):
          break
        psiProbs = self.getPossiblePhiAndProbs(trajs)
        hValue = 0
        for psi, prob in psiProbs:
          bins = [0] * k
          for idx in xrange(rewardCandNum):
            if psi[idx] > 0:
              # put opt policies into bins
              bins = [sum(_) for _ in zip(bins, policyBins[idx])]
          hValue += prob * scipy.stats.entropy(bins)
        if config.DEBUG: print hValue, psiProbs
        
        hValues[(s, indices)] += hValue
    
    minH = min(hValues.values())
    minStatesIndices = filter(lambda _: hValues[_] == minH, hValues.keys())
    minState, minIndices = random.choice(minStatesIndices)
    trajs = [self.sampleTrajFromRewardCandidate(idx, minState) for idx in minIndices]
    return trajs, None


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

    for s in args['S']:
      indices = numpy.random.choice(range(rewardCandNum), k, replace=False)
      for sampleIdx in range(config.SAMPLES_TIMES):
        trajs = [self.sampleTrajFromRewardCandidate(idx, s) for idx in indices]
        if any(len(u) < config.TRAJECTORY_LENGTH for u in trajs):
          assert sampleIdx == 0
          break
     
        for i in xrange(k):
          for j in xrange(k):
            hValues[(s, tuple(indices))] += self.cmp.getTrajectoryDistance(trajs[i], trajs[j])
    maxH = max(hValues.values())
    maxStatesIndices = filter(lambda _: hValues[_] == maxH, hValues.keys())
    maxState, maxIndices = random.choice(maxStatesIndices)
    trajs = [self.sampleTrajFromRewardCandidate(idx, maxState) for idx in maxIndices]
    return trajs, None


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

    for s in args['S']:
      indices = numpy.random.choice(range(rewardCandNum), k, replace=False)
      for sampleIdx in range(config.SAMPLES_TIMES):
        trajs = [self.sampleTrajFromRewardCandidate(idx, s) for idx in indices]
        if any(len(u) < config.TRAJECTORY_LENGTH for u in trajs):
          # this state is too close to the terminal state. not considering generating traj queries from here
          assert sampleIdx == 0
          break

        # compute the different between new psi and old psi
        psiProbs = self.getPossiblePhiAndProbs(trajs)
        for psi, prob in psiProbs:
          # note that we need to keep the information of which state to generate queries
          # and what reward candidates the policies are optimazing
          hValues[(s, tuple(indices))] += prob * sum(abs(p1 - p2) for p1, p2 in zip(psi, self.phi))

    maxH = max(hValues.values())
    maxStatesIndices = filter(lambda _: hValues[_] == maxH, hValues.keys())
    maxState, maxIndices = random.choice(maxStatesIndices)
    trajs = [self.sampleTrajFromRewardCandidate(idx, maxState) for idx in maxIndices]
    return trajs, None


class RandomTrajAgent(QTPAgent):
  def learn(self):
    args = easyDomains.convert(self.cmp, self.rewardSet, self.phi)
    k = config.NUMBER_OF_RESPONSES

    # find an arbitrary state to generate trajectory queries
    # make sure that the length of the query is TRAJECTORY_LENGTH
    while True:
      s = random.choice(args['S'])
      q = [tuple(self.sampleTrajectory(None, s, hori=config.TRAJECTORY_LENGTH, to='trajectory')) for _ in xrange(k)]
      if any(len(u) < config.TRAJECTORY_LENGTH for u in q): continue
      else: break
    return q, None
