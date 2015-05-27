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

Experiments
--------------

Experiments are written in shell scripts, which run some python files
sequentially.

Related files:
- [config.py](config.py) Some global configurations. This is very likely to be
  the only file you want to modify if you just want to run the existing
  experiments.
- [moduleTraining.sh](moduleTraining.sh) Code to train target, obstacle, path
  modules using q learning.
- [humanModularIRLExperiment.sh](humanModularIRLExperiment.sh) Please see inline
  comments for details. To run this, you need to have the human data {miniRes25.mat,
  subj[25-28].parsed.mat} in your checked-out folder, which are not included in
  this repo.

Learning Agents
--------------

Related files:

- [qlearningAgents.py](qlearningAgents.py) Q learning agent.
- [valueIterationAgents.py](valueIterationAgents.py) Value iteration learning agent.
- [modularAgents.py](modularAgents.py) Modular agent, derived from Q learning
  agent.

Environments
--------------

### Discrete domains

Related files:
- [gridworld.py](gridworld.py) Definition of grid MDP.
- [gridworldMaps.py](gridworldMaps.py) Specification of grid configurations.

Discrete environments are used to do some basic evaluation for modular RL
algorithms. The locations of objects and the agent are represented by grid
positions.

Examples for discrete environments can be found in gridworld.py, such as
`getMazeGrid`, `getBookGrid`, `getBridgeGrid`, etc.  You may add you own by
looking at their definitions. Concretely,

- define a get$GridlWorldName$ function in gridworldMaps.py. In which,
- create a two-dimensional list, usually called `grid`, specified as follows.

  * 'S': starting state. Multiple starting states can exist.
  * '#': obstacles that the agent cannot reach, i.e. P('#'|s, a) = 0.
  * A number: the reward upon reaching this state.

- create a lambda expression, usually called `isFinal`, to decide whether a
  state is a terminal state. If you want the task terminates upon receiving any
  reward, use `terminateIfInt`
- return `Gridworld(grid, isFinal)`

### Continuous domains

Related files:
- [continuousWorld.py](continuousWorld.py) The continuous world domain.
- [continuousWorldDomains.py](continuousWorldDomains.py) Concrete continuous
  world domain configurations. VR domains are parsed here.
- [humanWorld.py](humanWorld.py) The human world domain, derived from
  continuousWorld.

ContinuousWorld is a world that objects are represented by continuous
coordinates, as opposed to discrete domains where objects are located in grid
positions. The agent can move in eight directions (every 45 degrees). The
objects are targets, obstacles and path waypoints. These are parsed from
miniRes25.mat.

HumanWorld is revised from ContinuousWorld. The major difference is that the
agent has an orientation and can only go straight ahead, turn slightly left or
right.

Bayesian IRL
--------------

Related files:

- [inverseBayesianRL.py](inverseBayesianRL.py) implementation of the Bayesian IRL algorithm.
- [inverseBayesianRLExperiments.py](inverseBayesianRLExperiments.py) to run IRL on gridworld domains.

Implemented according to

Ramachandran, Deepak, and Eyal Amir.
"Bayesian inverse reinforcement learning."
Urbana 51 (2007): 61801.

Modular IRL
--------------

Related files:

- [inverseModularRL.py](inverseModularRL.py) implementation of the modular IRL algorithm.
- [inverseModularRLExperiments.py](inverseModularRLExperiments.py) to run IRL on humanWorld domains.
- [inverseModularRLTest.py](inverseModularRLTest.py) unit tests.

Inverse modular reinforcement learning is basically implemented according to

C. A. Rothkopf and Ballard, D. H.(2013), Modular inverse reinforcement
learning for visuomotor behavior, Biological Cybernetics,
107(4),477-490
http://www.cs.utexas.edu/~dana/Biol_Cyber.pdf

[Place holder: details of the algorithm will be described in my thesis]
