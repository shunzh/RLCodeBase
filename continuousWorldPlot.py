from graphics import *

class Plotting:
  def __init__(self, mdp, dim = 800):
    self.mdp = mdp
    self.size = max(mdp.xBoundary[1] - mdp.xBoundary[0], mdp.yBoundary[1] - mdp.yBoundary[0])
    self.radius = mdp.radius / self.size * dim
    self.dim = dim
    # for plot human trajectory, keep the previous state here
    self.prevState = None

    def shift(loc):
      """
      shift to the scale of the GraphWin
      """
      return (1.0 * (loc[0] - mdp.xBoundary[0]) / self.size * dim, 1.0 * (loc[1] - mdp.yBoundary[0]) / self.size * dim)

    self.shift = shift
 
  def drawDomain(self):
    """
    Args:
      mdp: parsed from mat file.

    Return:
      win object
    """
    self.win = GraphWin('Domain', self.dim, self.dim) # give title and dimensions

    rect = Rectangle(Point(0, 0), Point(self.dim, self.dim))
    rect.setFill('grey')
    rect.draw(self.win)
   
    def drawObjects(label, color):
      """
      Plot the objects as separate dots.
      """
      for obj in self.mdp.objs[label]:
        # graphically increase the radius
        cir = Circle(Point(self.shift(obj)), self.radius)
        cir.setFill(color)
        cir.draw(self.win)

    def drawSegments(label, color):
      """
      Plot the adjacent objects as segments.
      """
      prevObj = None
      for obj in self.mdp.objs[label]:
        if prevObj:
          # draw segments between waypoints
          line = Line(Point(self.shift(prevObj)), Point(self.shift(obj)))
          line.setWidth(3)
          line.setFill(color)
          line.draw(self.win)

          # draw a small circle at the waypoints
          cir = Circle(Point(self.shift(prevObj)), 5)
          cir.setFill(color)
          cir.draw(self.win)
        prevObj = obj
      
    targColor = color_rgb(80, 77, 157)
    obstColor = color_rgb(153, 77, 79)
    pathColor = color_rgb(200, 200, 200)
    drawObjects('targs', targColor)
    drawObjects('obsts', obstColor)
    drawSegments('segs', pathColor)
    drawObjects('elevators', pathColor)

    return self.win

  def plotHumanPath(self, x):
    # display the corresponding state in graphics
    if self.prevState != None:
      # only draw lines, so ignore the first state
      loc, orient = self.prevState
      newLoc, newOrient = x

      line = Line(Point(self.shift(loc)), Point(self.shift(newLoc)))
      line.setWidth(5)
      line.setFill(color_rgb(0, 255, 0))
      line.draw(self.win)

    self.prevState = x
