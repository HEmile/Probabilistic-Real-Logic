# TRAINING
MAX_TRAINING_ITERATIONS = 500

MAX_PRIOR_TRAINING_IT = 300

FREQ_OF_FEED_DICT_GENERATION = 10

SATURATION_LIMIT = 0.95

NOISE_VALUES = [0.]

N_POS_EXAMPLES_TYPES = 250
N_NEG_EXAMPLES_TYPES = 250
N_POS_EXAMPLES_PARTOF = 250
N_NEG_EXAMPLES_PARTOF = 250
number_of_pairs_for_axioms = 1000

# Options: 'all', 'indoor', 'vehicle', 'animal'
DATASET = 'indoor'

# List. Return [True, False] to create models for both training with and without constraints.
ALGORITHMS = ['prior', 'wc', 'nc']

# LOGIC TENSOR NETWORK SETUP
DEFAULT_LAYERS = 2

TYPE_LAYERS = 5

PART_OF_LAYERS = 2

REGULARIZATION = 1e-5

LAMBDA_2 = 1e-3

TNORM = "product"

FORALL_AGGREGATOR = "hmean"

POSITIVE_FACT_PENALTY = 0.

CLAUSE_AGGREGATOR = "log-likelihood"

OPTIMIZER = 'rmsprop'