import easyDomains
from consQueryAgents import ConsQueryAgent

LOCATION = 0
BOX1 = 1
BOX2 = 2
BOX3 = 3
DOOR1 = 4
DOOR2 = 5
SWITCH = 6

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
  sSets = [[(x, y) for x in range(width) for y in range(height)]]
  cIndices = range(1, len(sSets)) # location is not a constraint

  #aSets = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1),
  aSets = [(1, 0), (0, 1)]
       #'openDoor', #'closeDoor',
       #'turnOffSwitch']
  
  def move(s, a):  
    loc = s[LOCATION]
    if type(a) == tuple:
      sp = (loc[0] + a[0], loc[1] + a[1])
      if sp[0] >= 0 and sp[0] < width and sp[1] >= 0 and sp[1] < height:
        return sp
        # so it's not out of the border
        #if not (s[DOOR1] == CLOSED and sp == door1 or s[DOOR2] == CLOSED and sp == door2):
          # doors are fine
      else:
        return 'terminal'
    return loc
  
  def stepOnBoxGen(idx, box):
    def stepOnBox(s, a):
      loc = s[LOCATION]
      if loc == box or s[idx] == STEPPED: return STEPPED
      else: return CLEAN
    return stepOnBox
  
  def doorOpGen(idx, door):
    def doorOp(s, a):
      loc = s[LOCATION]
      doorState = s[idx]
      if loc == door:
        if doorState == OPEN and a == 'closeDoor': doorState = CLOSED
        elif doorState == CLOSED and a == 'openDoor': doorState = OPEN
      return doorState
    return doorOp
  
  def switchOp(s, a):
    loc = s[LOCATION]
    switchState = s[SWITCH]
    if loc == switch and a == 'turnOffSwitch': switchState = OFF 
    return switchState

  tFunc = [move]
#           stepOnBoxGen(BOX1, box1), stepOnBoxGen(BOX2, box2), stepOnBoxGen(BOX3, box3),
#           doorOpGen(DOOR1, door1), doorOpGen(DOOR2, door2),
#           switchOp]

  s0 = ((0, 0),) # robot's location
#       CLEAN, CLEAN, CLEAN, # boxes are clean
#       OPEN, CLOSED, # door 1 is open
#       1) # switch is on
  isTerminal = lambda s: s[0] == (width - 1, height - 1) # switch is off
  
  # there is a reward of -1 at any step except when goal is reached
  rFunc = lambda s, a: 10 if isTerminal(s) and a == (0, 0) else -1

  # the domain handler
  officeNav = easyDomains.getFactoredMDP(sSets, aSets, rFunc, tFunc, s0)
  print officeNav
  agent = ConsQueryAgent(officeNav, cIndices)
  print agent.findIrrelevantFeats()

if __name__ == '__main__':
  main()
