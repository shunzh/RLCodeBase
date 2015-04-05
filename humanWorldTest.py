"""
Test for humanWorld.
There is some non-intuitive computation involving distance and angle to an object
after an action. It needs to be checked.
""" 
import numpy as np
import unittest

import humanWorld

class Test(unittest.TestCase):
  def setUp(self):
    unittest.TestCase.setUp(self)
    self.transit = humanWorld.HumanWorld.transitionSimulate
    self.walkDist = 0.3
    self.turnDist = 0.3 * 0.25
    self.turnAngle = 30.0 / 180 * np.pi
  
  def test_parameter_consistency(self):
    """
    Just make sure that the parameters of the transitions are unchanged.
    In these tests, we check the output of the transition with directly computed results. 
    """
    self.assertAlmostEqual(humanWorld.HumanWorld.walkDist, self.walkDist)
    self.assertAlmostEqual(humanWorld.HumanWorld.turnDist, self.turnDist)
    self.assertAlmostEqual(humanWorld.HumanWorld.turnAngle, self.turnAngle)

  def test_transition_go(self):
    s = self.transit((self.walkDist + 0.1, 0), 'G')
    self.check_result(s, (0.1, 0))
  
  def test_transition_left(self):
    s = self.transit((self.turnDist + 0.1, -self.turnAngle), 'L')
    self.check_result(s, (0.1, 0))
  
  def test_transition_right(self):
    s = self.transit((self.turnDist + 0.1, self.turnAngle), 'R')
    self.check_result(s, (0.1, 0))
  
  def test_transition_complex_1(self):
    s = self.transit((self.walkDist * 2, self.turnAngle * 2), 'R')
    self.check_result(s, (0.536361, 34.0 / 180 * np.pi))
  
  def test_transition_complex_2(self):
    s = self.transit((self.walkDist * 2, self.turnAngle * 2), 'G')
    self.check_result(s, (0.519616, 90.0 / 180 * np.pi))
  
  def test_transition_complex_3(self):
    s = self.transit((self.walkDist * 2, self.turnAngle * 2), 'L')
    self.check_result(s, (0.60467, 97.1252 / 180 * np.pi))
  
  def check_result(self, output, desired):
    self.assertAlmostEqual(output[0], desired[0], places = 3, msg = 'Wrong distance ' + str(output[0]) + ' compared to ' + str(desired[0]))
    self.assertAlmostEqual(output[1], desired[1], places = 3, msg = 'Wrong orient ' + str(output[1]) + ' compared to ' + str(desired[1]))
    
if __name__ == '__main__':
  print "DUMMY FOR NOW. haven't updated"
  #unittest.main()