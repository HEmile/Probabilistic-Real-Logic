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
FREQ_OF_SAVE = 10005

# How often to print the current training loss
FREQ_OF_PRINT = 50

# How often to run the algorithms on the test set for monitoring on tensorboard. This is quite expensive.
FREQ_OF_TEST = 100

# Set to True to output the gradients and information about it in the console during testing. There are many!
PRINT_GRAD_DEBUG = False

# At what saturation (probability of knowledge base) to switch the feed dict. If a new feed dict is fed every iteration,
# this does nothing.
SATURATION_LIMIT = 0.95

# The amount of noises added to the data. This is a list: Every model is trained for every nose ratio.
# NOISE_VALUES = [0., 0.1, 0.2, 0.3, 0.4]
NOISE_VALUES = [0.]

# This is essentially the mini-batch size. The positive examples are for each type!
N_EXAMPLES_TYPES = 150
N_POS_EXAMPLES_PARTOF = 80
N_NEG_EXAMPLES_PARTOF = 80
number_of_pairs_for_axioms = 400

# Amount of pairs of bounding boxes to use for testing the validity of the rules
NUMBER_PAIRS_AXIOMS_TESTING = 2000

# The ratio of the split between labeled and unlabeled training data.
RATIO_DATA = [0.003]

# Set to False to just train on the complete dataset.
DO_SEMI_SUPERVISED = True

# Which types of the data to use. The paper is written using indoor. Options: 'all', 'indoor', 'vehicle', 'animal'
DATASET = 'indoor'

# List. The algorithms we wish to test during this evaluation. Choices: nc (no rules), wc (with rules), prior (deprecated)
ALGORITHMS = ['wc']

# Used to choose the split of the data
RANDOM_SEED = 1300

# Rerun the current setup this amount of times. The seeds used are RANDOM_SEED to RANDOM_SEED + AMOUNT_OF_EVALUATIONS
AMOUNT_OF_EVALUATIONS = 20

# Output name used in experiments
EXPERIMENT_NAME = 'no_MT_rerun'

# Used for preventing NaN's in some computations
EPSILON = 0.000001

# (Deprecated) The values of Lambda_2 used in experimenting with the informative prior. List.
LAMBDA_2_VALUES = [1e-5]

####################################
# LOGIC TENSOR NETWORK SETUP
####################################
# Default size of LTN for the softmax output/mutual exclusive predicates. Used for the type predicate.
MUT_EXCL_LAYERS = 10
# Default size of LTN for normal boolean predicates
DEFAULT_LAYERS = 2
# Size of LTN for the partof predicate.
PART_OF_LAYERS = 2

# Amount of regularization of the parameters
REGULARIZATION = 1e-8

# Semantics of 'or'. Choices: product, yager2, luk, goedel
TNORM = "product"
# Semantics of 'and'. Choices: product, goedel
SNORM = "product"
# Semantics of forall. Choices: product, mean-log-likelihood (averaged product), mean, gmean, hmean, min
FORALL_AGGREGATOR = "mean-log-likelihood"

# How to combine truth value of different rules/clauses. Choices: w-log-likelihood, min, mean, hmean, wmean (weighted),
# log-likelihood
CLAUSE_AGGREGATOR = "w-log-likelihood"

# Choices: rmsprop, ftrl, gd, ada, adam
OPTIMIZER = 'rmsprop'

# Weight the importance of different elements of the loss.
WEIGHT_ONTOLOGY_CLAUSES = 1.0
WEIGHT_LOGICAL_CLAUSES = 1.

WEIGHT_POS_PARTOF_EXAMPLES = 1.
WEIGHT_NEG_PARTOF_EXAMPLES = 1.
WEIGHT_TYPES_EXAMPLES = 2.

# Swaps the weight of the ontology clauses after some amount of training iterations. Currently unused.
ITERATIONS_UNTIL_WEIGHT_SWAP = 200000
WEIGHT_ONTOLOGY_CLAUSES_START = 1.
WEIGHT_ONTOLOGY_CLAUSES_END = 4.

# Stops the gradients in p(x, y) clauses of the form \forall x, y: p(x, y) -> a(x, y)
CAN_ONTOLOGY_TRAIN_PRECEDENT = True

# This didn't really seem to matter
CHEAT_SEMI_SUPERVISED = False

# Adds a layer that blocks gradients updating truth values that are already confident.
# The reasoning behind this is to reduce overconfidence of possibly wrong predictions.
USE_CLAUSE_FILTERING = False
# Truth value above which to filter gradients.
CLAUSE_FILTER_THRESHOLD = 0.8

# Instead of a hard cutoff in filtering, use a smooth filter given as exp(-x/SMOOTH_FILTER_FREQ), where the frequency
# is the truth value to maximize. 1 is an obvious choice.
USE_SMOOTH_FILTERING = False
SMOOTH_FILTER_FREQ = 1.0

USE_IMPLICATION_CLAUSES = True

# Normalizes the magnitude of the Modus Ponens and Modus Tollens gradients as explained in the paper.
NORMALIZE_PONENS_TOLLENS = False

# Stops gradients from flowing into p(x, y) in clauses of the form p(x, y) -> q(x, y). Not recommended
# in combination with the previous option.
STOP_MODUS_TOLLENS_UPDATES = True

# (Untested). Adds a penalty to the loss function for each positive literal in the batch.
POSITIVE_FACT_PENALTY = 0.

# (Unused) Regularization of prior mean
LAMBDA_2 = 1e-7

# (Untested) Can be disabled to used sigmoid outputs for the types. This is not recommended.
USE_MUTUAL_EXCL_PREDICATES = True
#################
# EVALUATION
# Note: evaluate.py is deprecated. Evaluation is done using tensorboard.
#################
THRESHOLDS = np.arange(.00, 1.1, .05)

EVAL_ALGORITHMS = ['prior_l2_0.001', 'prior_l2_0.0001', 'prior_l2_1e-06', 'wc','nc']

