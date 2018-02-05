# This is a very simple implementation of the UCT Monte Carlo Tree Search algorithm in Python 2.7.
# The function UCT(rootstate, itermax, verbose = False) is towards the bottom of the code.
# It aims to have the clearest and simplest possible code, and for the sake of clarity, the code
# is orders of magnitude less efficient than it could be made, particularly by using a 
# state.GetRandomMove() or state.DoRandomRollout() function.
# 
# Example GameState classes for Nim, OXO and Othello are included to give some idea of how you
# can write your own GameState use UCT in your 2-player game. Change the game to be played in 
# the UCTPlayGame() function at the bottom of the code.
# 
# Written by Peter Cowling, Ed Powley, Daniel Whitehouse (University of York, UK) September 2012.
# 
# Licence is granted to freely use and distribute for any sensible/legal purpose so long as this comment
# remains in any distributed code.
# 
# For more information about Monte Carlo Tree Search check out our web site at www.mcts.ai

from math import *
import random
import util

class Node:
    """ A node in the game tree. Note wins is always from the viewpoint of playerJustMoved.
        Crashes if state not specified.
    """
    def __init__(self, acts, checkConstraints, move = None, parent = None, state = None):
        self.move = move # the move that got us to this node - "None" for the root node
        self.parentNode = parent # "None" for the root node
        self.childNodes = []
        self.retSum = 0
        self.visits = 0
        
        self.acts = acts
        self.checkConstraints = checkConstraints
        self.untriedMoves = acts # future child nodes
        
    def UCTSelectChild(self):
        """ Use the UCB1 formula to select a child node. Often a constant UCTK is applied so we have
            lambda c: c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits to vary the amount of
            exploration versus exploitation.
        """
        s = sorted(self.childNodes, key = lambda c: c.retSum / c.visits + sqrt(2*log(self.visits)/c.visits))[-1]
        return s
    
    def AddChild(self, m, s):
        """ Remove m from untriedMoves and add a new child node for this move.
            Return the added child node
        """
        n = Node(self.acts, self.checkConstraints, move = m, parent = self, state = s)
        self.untriedMoves.remove(m)
        self.childNodes.append(n)
        return n
    
    def Update(self, result):
        """ Update this node - one additional visit and result additional wins. result must be from the viewpoint of playerJustmoved.
        """
        self.visits += 1
        self.retSum += result

    def __repr__(self):
        return "[M:" + str(self.move) + " W/V:" + str(self.wins) + "/" + str(self.visits) + " U:" + str(self.untriedMoves) + "]"

    def TreeToString(self, indent):
        s = self.IndentString(indent) + str(self)
        for c in self.childNodes:
            s += c.TreeToString(indent+1)
        return s

    def IndentString(self,indent):
        s = "\n"
        for i in range (1,indent+1):
            s += "| "
        return s

    def ChildrenToString(self):
        s = ""
        for c in self.childNodes:
            s += str(c) + "\n"
        return s


class MCTS:
  def __init__(self, S, A, r, T, s0, terminal, gamma=1, zeroConstraints=[]):
    """ Play a sample game between two UCT players where each player gets a different number 
        of UCT iterations (= simulations = tree nodes).
    """
    self.S = S
    self.A = A
    self.r = r
    # note that T is overriden so it finds the next state given s, a
    #TODO overfitting deterministic case
    self.move = lambda s, a: T(s, a)

    self.terminal = terminal
    self.s0 = s0
    self.gamma = gamma
    
    self.cons = zeroConstraints
    self.checkFeasible = lambda s, a: any((T(s, a), ap) in zeroConstraints for ap in A)

  def plan(self):
    x = util.Counter()
    s = self.s0
    while not self.terminal(s):
        a = self.UCT(s, itermax = 100, verbose = False)
        
        x[s, a] = 1 # record trajectory
        print s, a

        s = self.move(s, a)
    
    return x

  def UCT(self, rootstate, itermax, verbose = False):
      rootnode = Node(self.A, self.checkFeasible, state = rootstate)

      for i in range(itermax):
          node = rootnode
          state = rootstate.Clone()

          # Select
          while node.untriedMoves == [] and node.childNodes != []: # node is fully expanded and non-terminal
              node = node.UCTSelectChild()
              state = self.move(state, node.move)

          # Expand
          if node.untriedMoves != []: # if we can expand (i.e. state/node is non-terminal)
              m = random.choice(node.untriedMoves) 
              state = self.move(state, node.move)
              node = node.AddChild(m,state) # add child and descend tree

          # Rollout - this can often be made orders of magnitude quicker using a state.GetRandomMove() function
          while not self.terminal(state): # while state is non-terminal
              state = self.move(state, random.choice(self.A))

          # Backpropagate
          value = 0
          while node.parentNode != None: # backpropagate from the expanded node and work back to the root node
              m = node.move
              node = node.parentNode
              value += self.r(node.state, m) + self.gamma * value
              node.Update(value) # state is terminal. Update node with result from POV of node.playerJustMoved

      # Output some information about the tree - can be omitted
      if (verbose): print rootnode.TreeToString(0)
      else: print rootnode.ChildrenToString()

      return sorted(rootnode.childNodes, key = lambda c: c.visits)[-1].move # return the move that was most visited

