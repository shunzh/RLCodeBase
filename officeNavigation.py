import easyDomains
from consQueryAgents import ConsQueryAgent

LOCATION = 0
BOX1 = 1
BOX2 = 2
BOX3 = 3
DOOR1 = 1#4
DOOR2 = 2#5
SWITCH = 3#6

OPEN = 1
CLOSED = 0

STEPPED = 1
CLEAN = 0

ON = 1
OFF = 0 

def main():
  # specify the size of the domain, which are the robot's possible locations
  width = 5
  height = 3
  
  # some objects
  box1 = (2, 0)
  box2 = (0, 2)
  box3 = (2, 1)
  door1 = (1, 1)
  door2 = (3, 1)
  switch = (4, 2)
  
  # location, box1, box2, door1, door2, carpet, switch
  sSets = [[(x, y) for x in range(width) for y in range(height)],
           #[0, 1], [0, 1], [0, 1], #boxes
           [0, 1], [0, 1], #doors
           [0, 1]] #switch
  
  # the robot can change its locations and manipulate the switch
  cIndices = range(1, len(sSets) - 1) # location is not a constraint

  aSets = [(0, 0), (1, 0), (0, 1),#(-1, 0), (0, -1),
           'openDoor', 'closeDoor',
           'turnOffSwitch']

 
  def move(s, a):  
    loc = s[LOCATION]
    if type(a) == tuple:
      sp = (loc[0] + a[0], loc[1] + a[1])
      if sp[0] >= 0 and sp[0] < width and sp[1] >= 0 and sp[1] < height:
        # so it's not out of the border
        if not (s[DOOR1] == CLOSED and sp == door1 or s[DOOR2] == CLOSED and sp == door2):
          # doors are fine
          return sp
    return loc
  
  def stepOnBoxGen(idx, box):
    def stepOnBox(s, a):
      loc = s[LOCATION]
      boxState = s[idx]
      if loc == box: return STEPPED
      else: return boxState
    return stepOnBox
  
  def doorOpGen(idx, door):
    def doorOp(s, a):
      loc = s[LOCATION]
      doorState = s[idx]
      if loc in [(door[0] - 1, door[1]), (door[0], door[1])]:
        if a == 'closeDoor': doorState = CLOSED
        elif a == 'openDoor': doorState = OPEN
        # otherwise the door state is unchanged
      return doorState
    return doorOp
  
  def switchOp(s, a):
    loc = s[LOCATION]
    switchState = s[SWITCH]
    if loc == switch and a == 'turnOffSwitch': switchState = OFF 
    return switchState

  tFunc = [move,
           #stepOnBoxGen(BOX1, box1), stepOnBoxGen(BOX2, box2), stepOnBoxGen(BOX3, box3),
           doorOpGen(DOOR1, door1), doorOpGen(DOOR2, door2),
           switchOp]

  s0 = ((0, 0), # robot's location
        #CLEAN, CLEAN, CLEAN, # boxes are clean
        OPEN, CLOSED, # door 1 is open
        ON) # switch is on
  isTerminal = lambda s: s[SWITCH] == OFF # switch is off
  #isTerminal = lambda s: s[LOCATION] == (width - 1, height - 1) # top right corner
  
  # there is a reward of -1 at any step except when goal is reached
  rFunc = lambda s, a: 1 if isTerminal(s) else 0
  
  gamma = .9

  # the domain handler
  officeNav = easyDomains.getFactoredMDP(sSets, aSets, rFunc, tFunc, s0, gamma)
  
  # sanity checks
  """
  print officeNav['T'](((0, 1), OPEN, CLOSED, ON), 'closeDoor', ((0, 1), CLOSED, CLOSED, ON))
  print officeNav['T'](((2, 1), OPEN, CLOSED, ON), 'openDoor', ((2, 1), OPEN, OPEN, ON))
  print officeNav['T'](((0, 1), OPEN, CLOSED, ON), (1, 0), ((1, 1), OPEN, CLOSED, ON))
  print officeNav['T'](((2, 1), OPEN, OPEN, ON), (1, 0), ((3, 1), OPEN, OPEN, ON))
  print officeNav['T'](((2, 1), OPEN, CLOSED, ON), (1, 0), ((2, 1), OPEN, CLOSED, ON))
  print officeNav['T'](((4, 2), OPEN, CLOSED, ON), 'turnOffSwitch', ((4, 2), OPEN, CLOSED, OFF))
  """

  agent = ConsQueryAgent(officeNav, cIndices)
  print agent.findIrrelevantFeats()

if __name__ == '__main__':
  main()
