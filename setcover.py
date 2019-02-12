def findHighestFrequencyElement(elements, sets, weight=lambda _: 1):
  """
  Here we want to use elements to cover sets.
  This function finds the element that appears in the most number of the sets.

  sets: [[elements in one sets] for all sets]
  weights: find the element with the maximum weighted frequency. unweighted by default
  """
  appearenceFreq = {}
  
  for e in elements:
    appearenceFreq[e] = sum(e * weight(e) in s for s in sets)
  
  # return the index of the element that has the most appearances
  return max(appearenceFreq.iteritems(), key=lambda _: _[1])[0]