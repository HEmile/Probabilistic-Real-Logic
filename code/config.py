import numpy as np
############################################
# TRAINING
############################################

# Amount of iterations used to train the model
MAX_TRAINING_ITERATIONS = 6000

# Amount of iterations used to train the informative prior. Only for alg = 'prior'
MAX_PRIOR_TRAINING_IT = 140

# How often to change the data that's used to compute the gradient. 1 is recommended for pure stochastic descent
FREQ_OF_FEED_DICT_GENERATION = 1

# How often to save the model to the file. This is an expensive operation so it is not recommended to do this often
FREQ_OF_SAVE = 1000

# How often to print the current training loss
FREQ_OF_PRINT = 20

FREQ_OF_TEST = 40

# At what saturation (probability of knowledge base) to switch the feed dict. If a new feed dict is fed every iteration,
# this does nothing.
SATURATION_LIMIT = 0.95

# The amount of noises added to the data. This is a list: Every model is trained for every nose ratio.
# NOISE_VALUES = [0., 0.1, 0.2, 0.3, 0.4]
NOISE_VALUES = [0.]

# The values of Lambda_2 used in the informative prior.
# LAMBDA_2_VALUES = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
LAMBDA_2_VALUES = [1e-4]

# This is essentially the mini-batch size.
N_POS_EXAMPLES_TYPES = 50
N_NEG_EXAMPLES_TYPES = 50
N_POS_EXAMPLES_PARTOF = 50
N_NEG_EXAMPLES_PARTOF = 50
number_of_pairs_for_axioms = 200

NUMBER_PAIRS_AXIOMS_TESTING = 2000

RATIO_DATA = [0.003]

# Set to True if we train the rules on the complete data, and not just on the data selected using RATIO_DATA.
# This amounts to semi-supervised training, as the labels of the data are not used in the rules.
DO_SEMI_SUPERVISED = True

# Options: 'all', 'indoor', 'vehicle', 'animal'
DATASET = 'indoor'

# List. Return [True, False] to create models for both training with and without constraints.
# ALGORITHMS = ['prior', 'nc', 'wc']
ALGORITHMS = ['wc', 'nc']

RANDOM_SEED = 1238

EPSILON = 0.000001

####################################
# LOGIC TENSOR NETWORK SETUP
####################################
USE_MUTUAL_EXCL_PREDICATES = True

MUT_EXCL_LAYERS = 50

DEFAULT_LAYERS = 2

TYPE_LAYERS = 5

PART_OF_LAYERS = 2

REGULARIZATION = 1e-8

LAMBDA_2 = 1e-7

TNORM = "product"

FORALL_AGGREGATOR = "hmean"

POSITIVE_FACT_PENALTY = 0.

CLAUSE_AGGREGATOR = "log-likelihood"

OPTIMIZER = 'rmsprop'

#################
# EVALUATION
#################
THRESHOLDS = np.arange(.00, 1.1, .05)

EVAL_ALGORITHMS = ['prior_l2_0.001', 'prior_l2_0.0001', 'prior_l2_1e-06', 'wc','nc']