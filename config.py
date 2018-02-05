DEBUG = False

VERBOSE = False

SAVE_TO_FILE = True

# what to print
PRINT = 'perf'

NUMBER_OF_QUERIES = 1
NUMBER_OF_RESPONSES = 3

# a parameter for k way domains
CONNECTION_TYPE = 'grid'

INIT_STATE_DISTANCE = None

TRAJECTORY_LENGTH = 10

POLICY_TYPE = 'softmax'
#POLICY_TYPE = 'linear'

PG_TYPE = 'pe' # policy evaluation
#PG_TYPE = 'mc' # Monte Carlo

SAMPLE_TIMES = 20

# keep configuration options for output
opts = ''

#METHOD = 'lp'
METHOD = 'mcts'

# define the range of weights be between [-WEIGHT_MAX_VALUE, WEIGHT_MAX_VALUE]
WEIGHT_MAX_VALUE = 1
DIMENSION = 3