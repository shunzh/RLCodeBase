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

# in training phase
TRAINING = False

# enable to add slight turns
SLIGHT_TURNS = True

#SOLVER = "CMA-ES"
#SOLVER = "DE"
SOLVER = "BFGS"

BUDGET_SIZES = [10, 20, 50, 100, 150, 200, 250, 300, 350, 400]