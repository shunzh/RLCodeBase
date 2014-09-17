Maze and Pacman environments for Modular MDP
==============

**Licensing Information: Please do not distribute or publish solutions to this
project. You are free to use and extend these projects for educational
purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html**

This is implemented as instructed in reinforcement.html. As this repo is modified, it may not work as expected as specified in the html file.

Make sure you have python-tk installed for GUI display. You may do it by `sudo apt-get install python-tk`.

Useful commands
--------------

See all possible arguments:

``python gridworld.py -h``

Example:

``python gridworld.py -a Modular -g WalkAvoidGrid -k 10 -q``

- `-a Modular`: use modular agent
- `-g ObstacleGrid`: in ObstacleGrid domain.
- `-k 10`: run 10 iterations.
- `-q`: quite running, not showing the learning process.

Define new agent
--------------

For existing agents:

- Q learning agent is defined in qlearningAgents.py.
- VI learning agent is defined in valueIterationAgents.py.
- Modular MDP agent is defined in modularAgents.py.

ModularAgent is derived from ApproximateQAgent - which is a useful base class for creating new types of agent.

Define new gridworld environment
--------------

There are many examples in gridworld.py, such as `getMazeGrid`, `getBookGrid`, `getBridgeGrid`, etc.
You may add you own by looking at their definitions. Concretely,

- define a get$GridlWorldName$ function in gridworld.py. In which,
- create a two-dimensional list, usually called `grid`, specified as follows.

  * 'S': starting state. Multiple starting states can exist.
  * '#': obstacles that the agent cannot reach, i.e. P('#'|s, a) = 0.
  * A number: the reward upon reaching this state.

- create a lambda expression, usually called `isFinal`, to decide whether a state is a terminal state. If you want the task terminates upon receiving any reward, use `terminateIfInt`
- return `Gridworld(grid, isFinal)`

Modular IRL
--------------
Please look at the main function of inverseModularRL.py. `scipy.optimize.minimize` is used.

TODO
--------------

- Create a hyper level MDP. The states are the features extracted from sub-MDPs.
	* features: std of the sub MDP, max-Q value.
