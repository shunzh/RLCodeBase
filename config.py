"""
Some useful global configurations.
"""

# verbose output if enabled
DEBUG = False

# the agent looks at two closest objects
TWO_OBJECTS = False

# use discrete q tables
# use potential based q functions if false
DISCRETE_Q = False

# enable to add slight turns
SLIGHT_TURNS = True

TRAINING = False

HUMAN_SUBJECTS = range(25, 29)

# TODO for newly parsed files
# taskId -> {filename: domainIds}
HUMAN_SOURCE_CONFIG = lambda taskId:\
  {"subj" + str(num) + ".parsed.mat":\
   [range(0, 8), range(8, 16), range(16, 24), range(24, 32)][taskId]\
   for num in HUMAN_SUBJECTS}
OBSTACLE_RADIUS = 0

# budget for number of samples in IRL
BUDGET_SIZES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]