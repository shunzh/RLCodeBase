def findHighestFrequencyElement(newElements, addedElements, sets, weight=lambda _: 1):
  """
  Here we want to use elements to cover sets.
  This function finds the element that appears in the most number of the sets.

  sets: [[elements in one sets] for all sets]
  weights: find the element with the maximum weighted frequency. unweighted by default
  """
  # see what sets are uncovered by addedElements
  unCoveredSets = filter(lambda s: not any(e in s for e in addedElements), sets)
  # if they are all covered, we are done
  if len(unCoveredSets) == 0: return None

  appearenceFreq = {}
  
  for e in newElements:
    appearenceFreq[e] = weight(e) * sum(e in s for s in unCoveredSets)
  
  # return the index of the element that has the most appearances
  return max(appearenceFreq.iteritems(), key=lambda _: _[1])[0]