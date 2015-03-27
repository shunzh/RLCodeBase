Modular Reinforcement Learning Codebase
==============

**Licensing Information: Please do not distribute or publish solutions to this
project. You are free to use and extend these projects for educational
purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html**

Introduction
--------------

The goal of this codebase is to provide reinforcement learning, inverse
reinforcement learning methods, and domains for evaluation. The codebase is
written in Python. This is for the research in Prof. Dana Ballard's group at
the University of Texas at Austin.

Useful Commands
--------------

Make sure you have python-tk installed for GUI display. You may do it by running

``sudo apt-get install python-tk``

See all possible arguments:

``python humanWorld.py -h``

Example:

``python humanWorld.py -a q -g simple -c obsts -k 10 -q``

- `-a q`: use Q-learning agent
- `-g simple`: in the simple domain.
- `-k 10`: run 10 iterations.
- `-q`: quite running, not showing the learning process.

Learning Agents
--------------

For existing agents:

- Q learning agent is defined in qlearningAgents.py.
- VI learning agent is defined in valueIterationAgents.py.
- Modular Q agent is defined in modularAgents.py.

Environments
--------------

Naming conventions:

\*World.py specifies the domains.
\*WorldExperiment.py can be used to run the experiment on the corresponding domain.
\*WorldPlot.py contains graphic supports.
\*WorldTest.py contains unit tests. 
### Discrete domains

Examples for discrete environments can be found in gridworld.py, such as `getMazeGrid`, `getBookGrid`, `getBridgeGrid`, etc.
You may add you own by looking at their definitions. Concretely,

- define a get$GridlWorldName$ function in gridworld.py. In which,
- create a two-dimensional list, usually called `grid`, specified as follows.

  * 'S': starting state. Multiple starting states can exist.
  * '#': obstacles that the agent cannot reach, i.e. P('#'|s, a) = 0.
  * A number: the reward upon reaching this state.

- create a lambda expression, usually called `isFinal`, to decide whether a state is a terminal state. If you want the task terminates upon receiving any reward, use `terminateIfInt`
- return `Gridworld(grid, isFinal)`

### Continuous domains

For continuous environments, I have implemented `ContinuousWorld`, `humanWrold`. You can find in corresponding files.

Modular IRL
--------------

Related files:

- [inverseModularRL.py](inverseModularRL.py) implementation of the modular IRL algorithm.
- [inverseModularRLExperiments.py](inverseModularRLExperiments.py) to run IRL on humanWorld domains.
- [inverseModularRLTest.py](inverseModularRLTest.py) unit tests.

Inverse modular reinforcement learning is implemented according to

C. A. Rothkopf and Ballard, D. H.(2013), Modular inverse reinforcement
learning for visuomotor behavior, Biological Cybernetics,
107(4),477-490
http://www.cs.utexas.edu/~dana/Biol_Cyber.pdf

