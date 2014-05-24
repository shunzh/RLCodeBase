Maze and Pacman environment for Modular MDP
==============

**Licensing Information: Please do not distribute or publish solutions to this
project. You are free to use and extend these projects for educational
purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html**

This ia a subtree of 391L repo, which is HW5 in git@github.com:szhang90/391L.git.

Make sure you have python-tk installed for GUI display. Do it by `sudo apt-get install python-tk`.

Useful commands
--------------

See all possible arguments:

``python gridworld.py -h``

Example:

``python gridworld.py -a Modular -g ObstacleGrid -k 10``

The runs Modular agent in ObstacleGrid domain, with 10 iterations.

Define new gridworld environment
--------------

- define a get$GridlWorldName$ function in gridworld.py. In which,
- create a two-dimensional list, usually called `grid`, specified as follows.

'S': starting state. Multiple starting states can exist.
'#': obstacles that the agent cannot reach, i.e. P('#'|s, a) = 0.
A number: the reward upon reaching this state.

- create a lambda expression, usually called `isFinal`, to decide whether a state is a terminal state. If you want the task terminates upon receiving any reward, use `terminateIfInt`
- return `Gridworld(grid, isFinal)`
