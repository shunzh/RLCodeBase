def findHighestFrequencyElement(feats, sets, weight=lambda _: 1):
  """
  Here we want to use elements to cover sets.
  This function finds the element that appears in the most number of the sets.

  sets: [[elements in one sets] for all sets]
  weights: find the element with the maximum weighted frequency. unweighted by default
  """
  if len(sets) == 0: return None

  appearenceFreq = {}
  
  for e in feats:
    appearenceFreq[e] = weight(e) * sum(e in s for s in sets)
  
  # return the index of the element that has the most appearances
  return max(appearenceFreq.iteritems(), key=lambda _: _[1])[0]
  
def coverFeat(feat, sets):
  """
  Find the new set of sets if feat is covered.
  We only need to remove the sets that contain feat.
  """
  return filter(lambda s: feat not in s, sets)

def removeFeat(feat, sets):
  """
  Find the new set of sets if feat is removed.
  We remove feat, and remove sets that are reducible (which are supersets of any other set).
  """
  newSets = map(lambda s: set(s) - {feat}, sets)
  newSets = filter(lambda s: not any(otherSet.issubset(s) for otherSet in newSets if otherSet != s), newSets)
  return map(lambda s: tuple(s), newSets)