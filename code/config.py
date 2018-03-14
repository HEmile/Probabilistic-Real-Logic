############################################
# TRAINING
############################################

# Amount of iterations used to train the model
MAX_TRAINING_ITERATIONS = 500

# Amount of iterations used to train the informative prior. Only for alg = 'prior'
MAX_PRIOR_TRAINING_IT = 150

# How often to change the data that's used to compute the gradient. 1 is recommended for pure stochastic descent
FREQ_OF_FEED_DICT_GENERATION = 1

# How often to save the model to the file. This is an expensive operation so it is not recommended to do this often
FREQ_OF_SAVE = 1000

# How often to print the current training loss
FREQ_OF_PRINT = 20

# At what saturation (probability of knowledge base) to switch the feed dict. If a new feed dict is fed every iteration,
# this does nothing.
SATURATION_LIMIT = 0.95

# The amount of noises added to the data. This is a list: Every model is trained for every nose ratio.
# NOISE_VALUES = [0., 0.1, 0.2, 0.3, 0.4]
NOISE_VALUES = [0.]

# The values of Lambda_2 used in the informative prior.
LAMBDA_2_VALUES = [0.1, 0.01, 1e-3, 1e-4, 1e-5]

# This is essentially the mini-batch size.
N_POS_EXAMPLES_TYPES = 250
N_NEG_EXAMPLES_TYPES = 250
N_POS_EXAMPLES_PARTOF = 250
N_NEG_EXAMPLES_PARTOF = 250
number_of_pairs_for_axioms = 1000

RATIO_DATA = [int(1), 0.25, 0.01]

# Options: 'all', 'indoor', 'vehicle', 'animal'
DATASET = 'indoor'

# List. Return [True, False] to create models for both training with and without constraints.
# ALGORITHMS = ['prior']
ALGORITHMS = ['prior', 'nc', 'wc']

EVAL_ALGORITHMS = ['prior_l2_0.001', 'prior_l2_0.0001', 'wc','nc']
# EVAL_ALGORITHMS = ['prior_l2_0.001', 'prior_l2_1e-05', 'wc', 'nc']

# LOGIC TENSOR NETWORK SETUP
DEFAULT_LAYERS = 2

TYPE_LAYERS = 5

PART_OF_LAYERS = 2

REGULARIZATION = 1e-6

LAMBDA_2 = 1e-7

TNORM = "product"

FORALL_AGGREGATOR = "product"

POSITIVE_FACT_PENALTY = 0.

CLAUSE_AGGREGATOR = "log-likelihood"

OPTIMIZER = 'rmsprop'