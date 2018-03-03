# TRAINING
MAX_TRAINING_ITERATIONS = 500

FREQ_OF_FEED_DICT_GENERATION = 10

SATURATION_LIMIT = 0.95

NOISE_VALUES = [0., 0.1, 0.2, 0.3, 0.4]

# List. Return [True, False] to create models for both training with and without constraints.
WC_TRAIN = [True, False]

# LOGIC TENSOR NETWORK SETUP
LAYERS = 2

REGULARIZATION = 1e-15

TNORM = "luk"

FORALL_AGGREGATOR = "hmean"

POSITIVE_FACT_PENALTY = 0.

CLAUSE_AGGREGATOR = "log-likelihood"

OPTIMIZER = 'rmsprop'