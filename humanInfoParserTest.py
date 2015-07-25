"""
Test for humanInfoParser.
The parsed data should be correct if no error reported, but we need to check whether
human subjects do what they are supposed to in the experiments. 
"""
import unittest
import numpy as np
import humanInfoParser

class Test(unittest.TestCase):
  def setUp(self):
    unittest.TestCase.setUp(self)

    # threshold on proportion of actions inconsistent with the designated task. 
    self.threshold = 0.3

    subjFiles = ["subj" + str(num) + ".parsed.mat" for num in xrange(25, 29)]
    taskRanges = [range(0, 8), range(8, 16), range(16, 24), range(24, 31)]
    
    # range of angles for actions
    goAngle = 30.0 / 180 * np.pi
    turnAngle = 15.0 / 180 * np.pi
    
    persueBehavior = lambda s, a: abs(s[1]) < goAngle and a == 0 or\
                                  s[1] < -turnAngle and a < 0 or\
                                  s[1] > turnAngle and a > 0
    avoidBehavior  = lambda s, a: abs(s[1]) < goAngle and a == 0 or\
                                  s[1] < -turnAngle and a < 0 or\
                                  s[1] > turnAngle and a > 0

    consistentActions = [0, 0, 0]
    allActions = [0, 0, 0]
    
    for idx in xrange(len(taskRanges)):
      samples = humanInfoParser.getHumanStatesActions(subjFiles, taskRanges[idx])
      
      for sample in samples:
        # decouple
        state, action = sample
        targ, _, obst, _, seg, _ = state
        
        if idx == 0:
          # task 1: path only
          consistentActions[0] += persueBehavior(seg, action)
          allActions[0] += 1
        elif idx == 1:
          # task 2: obstacle avoidence, follow path
          consistentActions[1] += avoidBehavior(obst, action) or persueBehavior(seg, action)
          allActions[1] += 1
        elif idx == 2:
          # task 3: target getting, follow path
          consistentActions[2] += persueBehavior(targ, action) or persueBehavior(seg, action)
          allActions[2] += 1         
        # nothing to check for task 4 (idx 3)
    
    self.consistencies = [1.0 * consActs / allActs for consActs, allActs in zip(consistentActions, allActions)]
  
  def test_task_1(self):
    consistencies = self.consistencies
    self.assertGreaterEqual(consistencies[0], 1 - self.threshold, msg='task 1 bad consistency ' + str(consistencies[0]))
  
  def test_task_2(self):
    consistencies = self.consistencies
    self.assertGreaterEqual(consistencies[1], 1 - self.threshold, msg='task 2 bad consistency ' + str(consistencies[1]))

  def test_task_3(self):
    consistencies = self.consistencies
    self.assertGreaterEqual(consistencies[2], 1 - self.threshold, msg='task 3 bad consistency ' + str(consistencies[2]))
   
if __name__ == '__main__':
  unittest.main()