import sys
import tensorflow as tf
import numpy as np
import pdb
import config


def train_op(loss, optimization_algorithm):
    with tf.variable_scope('optimizer') as sc:
        if optimization_algorithm == "ftrl":
            optimizer = tf.train.FtrlOptimizer(learning_rate=0.01, learning_rate_power=-0.5)
        if optimization_algorithm == "gd":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
        if optimization_algorithm == "ada":
            optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
        if optimization_algorithm == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01, decay=0.9)
        if optimization_algorithm == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        return optimizer.minimize(loss)


def PR(tensor):
    np.set_printoptions(threshold=np.nan)
    return tf.Print(tensor, [tf.shape(tensor), tensor.name, tensor], summarize=200000)


def t_norm(literals, label="", tnorm=config.TNORM):
    with tf.variable_scope('t_norm_' + label) as sc:
        list_of_literal_tensors = [lit.tensor for lit in literals]
        literals_tensor = tf.concat(list_of_literal_tensors, 1)
        if tnorm == "product":
            result = 1.0 - tf.reduce_prod(1.0 - literals_tensor, 1, keep_dims=True)
        if tnorm == "yager2":
            result = tf.minimum(1.0, tf.sqrt(tf.reduce_sum(tf.square(literals_tensor), 1, keep_dims=True)))
        if tnorm == "luk":
            result = tf.minimum(1.0, tf.reduce_sum(literals_tensor, 1, keep_dims=True))
        if tnorm == "goedel":
            result = tf.reduce_max(literals_tensor, 1, keep_dims=True, name=label)
        return result

def s_norm(literals, label="", snorm=config.SNORM):
    with tf.variable_scope('s_norm' + label) as sc:
        list_of_literal_tensors = [lit.tensor for lit in literals]
        literals_tensor = tf.concat(list_of_literal_tensors, 1)
        if snorm == "product":
            result = tf.reduce_prod(literals_tensor, 1, keep_dims=True)
        if snorm == "goedel":
            result = tf.reduce_min(literals_tensor, 1, keep_dims=True, name=label)
        return result

def disjunction_of_literals(literals, label="no_label"):
    with tf.variable_scope('disjunction_' + label) as sc:
        result = t_norm(literals, label)

        if config.USE_CLAUSE_FILTERING:
            result = tf.where(tf.greater(config.CLAUSE_FILTER_THRESHOLD, result),
                              result, tf.clip_by_value(result, 1, 1), name='clause_filter')
            # result = tf.minimum(result, config.CLAUSE_FILTER_THRESHOLD, name='clause_filter')

        if config.FORALL_AGGREGATOR == "product":
            if config.CLAUSE_AGGREGATOR == 'log-likelihood':
                return tf.reduce_sum(tf.log(result + config.EPSILON), keep_dims=True, name=label)
            else:
                return tf.exp(tf.reduce_mean(tf.log(result + config.EPSILON), keep_dims=True), name=label)
        if config.FORALL_AGGREGATOR == "mean-log-likelihood":
            return tf.reduce_mean(tf.log(result + config.EPSILON), keep_dims=True, name=label)
        if config.FORALL_AGGREGATOR == "mean":
            return tf.reduce_mean(result, keep_dims=True, name=label)
        if config.FORALL_AGGREGATOR == "gmean":
            return tf.exp(tf.multiply(tf.reduce_sum(tf.log(result), keep_dims=True),
                                 tf.reciprocal(tf.to_float(tf.size(result)))), name=label)
        if config.FORALL_AGGREGATOR == "hmean":
            return tf.div(tf.to_float(tf.size(result)), tf.reduce_sum(tf.reciprocal(result), keep_dims=True))
        if config.FORALL_AGGREGATOR == "min":
            return tf.reduce_min(result, keep_dims=True, name=label)

def clause_filter(clauses):
    with tf.variable_scope('clause_filter') as sc:
        if config.USE_SMOOTH_FILTERING:
            return tf.multiply(clauses, tf.exp(-1 / config.SMOOTH_FILTER_FREQ * clauses))
        else:
            return tf.where(tf.greater(config.CLAUSE_FILTER_THRESHOLD, clauses),
                                   clauses, tf.clip_by_value(clauses, 1, 1))

# Domain defines a term-space. The domain is a subset of vectors in real^self.columns.
# self.tensor is assigned with a feed dict to actually instantiate the domain with objects.
# The size of the domain is not specified beforehand: You can feed any number of objects.
# Other parts of the code refer to Domains to iterate over those objects.
class Domain:
    # columns are a number: The amount of features used.
    def __init__(self, columns, dom_type="float", label=None):
        with tf.variable_scope('domain_' + label):
            self.columns = columns
            self.label = label
            self.tensor = tf.placeholder(dom_type, shape=[None, self.columns], name='placeholder')
            self.parameters = []


class Domain_concat(Domain):
    def __init__(self, domains):
        self.columns = np.sum([dom.columns for dom in domains])
        self.label = "concatenation of" + ",".join([dom.label for dom in domains])
        self.tensor = tf.concat([dom.tensor for dom in domains], 1)
        self.parameters = [par for dom in domains for par in dom.parameters]


class Domain_slice(Domain):
    def __init__(self, domain, begin_column, end_column):
        self.columns = end_column - begin_column
        self.label = "projection of" + domain.label + "from column " + begin_column + " to column " + end_column
        self.tensor = tf.concat(tf.split(1, domain.columns, domain.tensor)[begin_column:end_column], 1)
        self.parameters = domain.parameters

# I think this is used for function symbols. It is not used
# anywhere in the code base.
class Function(Domain):
    def __init__(self, label, domain, range, value=None):
        self.label = label
        self.domain = domain
        self.range = range
        self.value = value
        if self.value:
            self.parameters = []
        else:
            self.M = tf.Variable(tf.random_normal([self.domain.columns,
                                                   self.range.columns]),
                                 name="M_" + self.label)

            self.n = tf.Variable(tf.random_normal([1, self.range.columns]),
                                 name="n_" + self.label)
            self.parameters = [self.n, self.M]
        if self.value:
            self.tensor = self.value
        else:
            self.tensor = tf.add(tf.matmul(self.domain, self.M), self.n)


class Predicate:
    def __init__(self, label, domain, layers=config.DEFAULT_LAYERS):
        with tf.variable_scope('predicate_' + label) as sc:
            print(sc)
            self.domain = domain
            self.number_of_layers = layers
            self.W = tf.Variable(tf.random_normal([layers,
                                                   self.domain.columns,
                                                   self.domain.columns]),
                                 name="W")
            self.V = tf.Variable(tf.random_normal([layers,
                                                   self.domain.columns]),
                                 name="V")
            self.b = tf.Variable(-(tf.ones([1, layers])),
                                 name="b")
            self.u = tf.Variable(tf.ones([layers, 1]),
                                 name="u")
            self.parameters = [self.W, self.V, self.b, self.u]
            self.label = label

    # Here is where the logic tensor network magic happens. It creates a tensor
    # that takes as input the domain of the predicate and computes the value of
    # the predicate for all elements in the domain.
    def tensor(self, domain=None):
        with tf.variable_scope('predicate_' + self.label + domain.label) as sc:
            if domain is None:
                domain = self.domain
            X = domain.tensor
            XW = tf.matmul(tf.tile(tf.expand_dims(X, 0), [self.number_of_layers, 1, 1]), self.W)
            XWX = tf.squeeze(tf.matmul(tf.expand_dims(X, 1), tf.transpose(XW, [1, 2, 0])))
            XV = tf.matmul(X, tf.transpose(self.V))
            gX = tf.matmul(tf.tanh(XWX + XV + self.b), self.u)
            return tf.sigmoid(gX, name='predicate_output')


class _MutExclPredicate(Predicate):
    def __init__(self, label, domain, tensor, index):
        self.label = label
        self.domain = domain
        self._tensor = tensor
        self.parameters = []
        self.index = index

    def tensor(self, domain=None):
        return self._tensor(domain, self.index)


class MutualExclusivePredicates:
    def __init__(self, label, amt_predicates, domain, layers=config.DEFAULT_LAYERS):
        with tf.variable_scope('MutualExclusivePredicate_' + label) as sc:
            self.domain = domain
            self.label = label
            self.number_of_layers = layers
            self.amt_predicates = amt_predicates

            self.W = tf.Variable(tf.random_normal([layers,
                                                   self.domain.columns,
                                                   self.domain.columns]),
                                 name="W")
            self.V = tf.Variable(tf.random_normal([layers,
                                                   self.domain.columns]),
                                 name="V")
            self.b = tf.Variable(-(tf.ones([1, layers])),
                                 name="b" + label)
            self.U = tf.Variable(tf.ones([layers, amt_predicates]),
                                 name="U")
            self.parameters = [self.W, self.V, self.b, self.U]
            # Contains a tensor for each unique domain
            self.tensors = {}
            self.predicates = []
            for i in range(amt_predicates):
                self.predicates.append(_MutExclPredicate(label + str(i), self.domain, self.tensor, i))

    def tensor(self, domain, output_index):
        logits = self.logits(domain)
        softmax_layer = tf.nn.softmax(logits, name='output_' + (domain.label if domain else self.domain.label))
        return tf.reshape(softmax_layer[:, output_index], [-1, 1])

    def logits(self, domain):
        if domain is None:
            domain = self.domain
        if domain in self.tensors:
            logits = self.tensors[domain]
        else:
            with tf.variable_scope('MutualExclusivePredicate_' + self.label + domain.label) as sc:
                X = domain.tensor
                XW = tf.matmul(tf.tile(tf.expand_dims(X, 0), [self.number_of_layers, 1, 1]), self.W)
                XWX = tf.squeeze(tf.matmul(tf.expand_dims(X, 1), tf.transpose(XW, [1, 2, 0])))
                XV = tf.matmul(X, tf.transpose(self.V))
                logits = tf.matmul(tf.nn.relu(XWX + XV + self.b), self.U)
        return logits

class Literal:
    def __init__(self, polarity, predicate, domain=None, disable_gradient=False, name=''):
        self.predicate = predicate
        self.polarity = polarity
        self.domain = domain
        self.name = name
        if domain is None:
            self.domain = predicate.domain
        if polarity:
            self.tensor = predicate.tensor(domain)
        else:
            self.tensor = 1 - predicate.tensor(domain)
        if disable_gradient:
            self.tensor = tf.stop_gradient(self.tensor)

# Clauses are a disjunction of literals. Other forms of clauses are currently
# not accepted.
class Clause:
    def __init__(self, literals, label="", weight=1.0):
        with tf.variable_scope('Clause_' + label) as sc:
            self.weight = weight
            self.label = label
            self.literals = literals
            self.tensor = tf.reshape(disjunction_of_literals(self.literals, label=label), (), name='satisfaction')
            self.predicates = set([lit.predicate for lit in self.literals])

class TypeClause(Clause):
    def __init__(self, mut_excl_pred, domain=None, weight=1.0, label=''):
        with tf.variable_scope('TypeClause_' + label) as sc:
            self.weight = weight
            self.label = label
            self.correct_labels = tf.placeholder('int32', shape=[None], name='placeholder')
            self.tensor = tf.reduce_mean(-tf.nn.softmax_cross_entropy_with_logits\
                (labels=tf.one_hot(self.correct_labels, mut_excl_pred.amt_predicates),
                 logits=mut_excl_pred.logits(domain)), name='satisfaction')

class ImplicationClause(Clause):
    def __init__(self, antecedent, consequent, label="", weight=1.0):
        with tf.variable_scope('ImplicationClause_' + label) as sc:
            self.antecedent = antecedent
            self.consequent = consequent
            self.weight = weight
            self.label = label
            self.literals = antecedent + consequent
            self.predicates = set([lit.predicate for lit in self.literals])
            # Connect the antecedent through the s-norm (and).
            self.P = s_norm(antecedent, label + "antecedent")
            if consequent:
                self.Q = t_norm(consequent, label + "consequent")
            else:
                # This means the antecedent just has to be true
                self.Q = 0.0
            if config.STOP_MODUS_TOLLENS_UPDATES:
                self.P = tf.stop_gradient(self.P)

            # The implication is computed as not(p and not q) and thus follows the s-norm semantics
            if config.SNORM == 'product':
                self.truth_val = 1 - self.P * (1 - self.Q)
            if config.SNORM == 'goedel':
                self.truth_val = 1 - tf.minimum(self.P, 1 - self.Q)
            self.grad_mp = tf.div(self.P, self.truth_val)
            self.grad_mt = tf.div(1 - self.Q, self.truth_val)
            if config.NORMALIZE_PONENS_TOLLENS:
                # Assuming product norm
                self.not_P = 1 - self.P

                # self.grad_mp = 0.5 * tf.stop_gradient(tf.div(self.grad_mp, tf.reduce_sum(self.grad_mp)))
                # self.grad_mt = 0.5 * tf.stop_gradient(tf.div(self.grad_mt, tf.reduce_sum(self.grad_mt)))

                self.grad_mp = tf.div(self.grad_mp, tf.reduce_sum(self.grad_mp))
                self.grad_mt = tf.div(self.grad_mt, tf.reduce_sum(self.grad_mt))

                self.tensor = tf.reduce_sum(tf.multiply(self.grad_mp, self.Q) + tf.multiply(self.grad_mt, self.not_P))
            else:
                # Do not propagate gradients (Disable modus tollens)
                self.grad_mt = 0 * self.grad_mt
                # Connect the consequent through the t-norm (or)

                if config.USE_CLAUSE_FILTERING:
                    self.tensor = clause_filter(self.tensor)
                self.tensor = tf.reduce_mean(tf.log(self.truth_val + config.EPSILON), name='satisfaction')
            # self.tensor = tf.reduce_mean(tf.multiply(tf.log(self.con_tensor + config.EPSILON), self.ant_tensor), name='satisfaction')

class KnowledgeBase:
    # Note: This does not currently support functions
    def __init__(self, predicates, mutExPreds, clauses, kbLabel, save_path=""):
        with tf.variable_scope('KnowledgeBase' + kbLabel) as sc:
            self.clauses = clauses
            if not self.clauses:
                self.tensor = tf.constant(1.0)
            else:
                weights_tensor = tf.stack([cl.weight for cl in clauses])
                clauses_value_tensor = tf.stack([cl.tensor for cl in clauses], 0)
                if config.CLAUSE_AGGREGATOR == "min":
                    print("clauses aggregator is min")
                    self.tensor = tf.reduce_min(clauses_value_tensor)
                if config.CLAUSE_AGGREGATOR == "mean":
                    self.tensor = tf.reduce_mean(clauses_value_tensor)
                if config.CLAUSE_AGGREGATOR == "hmean":
                    self.tensor = tf.div(tf.to_float(tf.size(clauses_value_tensor)),
                                         tf.reduce_sum(tf.reciprocal(clauses_value_tensor), keep_dims=True))
                if config.CLAUSE_AGGREGATOR == "wmean":
                    self.tensor = tf.div(tf.reduce_sum(tf.multiply(weights_tensor, clauses_value_tensor)),
                                         tf.reduce_sum(weights_tensor))
                if config.CLAUSE_AGGREGATOR == 'log-likelihood':
                    # Smartly handle exp/log functions as it already uses exp sum log trick to compute product norm.
                    if config.FORALL_AGGREGATOR == 'product' or config.FORALL_AGGREGATOR == 'mean-log-likelihood':
                        self.tensor = tf.reduce_mean(clauses_value_tensor)
                    else:
                        self.tensor = tf.reduce_mean(tf.log(clauses_value_tensor))
                if config.CLAUSE_AGGREGATOR == 'w-log-likelihood':
                    # Smartly handle exp/log functions as it already uses exp sum log trick to compute product norm.
                    if config.FORALL_AGGREGATOR == 'product' or config.FORALL_AGGREGATOR == 'mean-log-likelihood':
                        self.tensor = tf.reduce_mean(tf.multiply(clauses_value_tensor, weights_tensor / tf.reduce_sum(weights_tensor)))
                    else:
                        self.tensor = tf.reduce_mean(tf.multiply(tf.log(clauses_value_tensor), weights_tensor))
            self.tensor = tf.reshape(self.tensor, shape=(), name=kbLabel + 'loss')
            tf.summary.scalar(kbLabel + 'loss', self.tensor, collections=['train'])

            self.parameters = [param
                               for pred in predicates
                               for param in pred.parameters]
            self.parameters += [param
                                for mutPred in mutExPreds
                                for param in mutPred.parameters]
            self.omega = tf.concat([tf.reshape(par, [-1]) for par in self.parameters], 0)
            self.omega = tf.reshape(self.omega, [-1])  # Completely flatten the parameter array
            self.num_params = tf.shape(self.omega)
            self.prior_mean = tf.placeholder("float", shape=[None,], name="prior_mean")
            self.prior_lambda = tf.placeholder("float", shape=(), name='prior_lambda')
            self.L2_regular = tf.reduce_sum(tf.square(self.omega - self.prior_mean)) * self.prior_lambda

            tf.summary.scalar(kbLabel + 'regularization', self.L2_regular, collections=['train'])

            if config.POSITIVE_FACT_PENALTY != 0:
                self.loss = self.L2_regular + \
                            tf.multiply(config.POSITIVE_FACT_PENALTY, self.penalize_positive_facts()) - \
                            PR(self.tensor)
            else:
                self.loss = self.L2_regular - self.tensor#PR(self.tensor)
            self.save_path = save_path
            self.train_op = train_op(self.loss, config.OPTIMIZER)

    def penalize_positive_facts(self):
        tensor_for_positive_facts = [tf.reduce_sum(Literal(True, lit.predicate, lit.domain).tensor, keep_dims=True) for
                                     cl in self.clauses for lit in cl.literals]
        return tf.reduce_sum(tf.concat(tensor_for_positive_facts, 0))

    def save(self, sess, label, version=""):
        self.saver = tf.train.Saver()
        print('Saving with label', label)
        save_path = self.saver.save(sess, self.save_path + label + version + ".ckpt")
        print('Saved to', save_path)

    def restore(self, sess):
        ckpt = tf.train.get_checkpoint_state(self.save_path)
        if ckpt and ckpt.model_checkpoint_path:
            print("restoring model")
            self.saver.restore(sess, ckpt.model_checkpoint_path)

    def train(self, sess, feed_dict):
        o, l, t, reg = sess.run([self.train_op, self.loss, self.tensor, self.L2_regular], feed_dict)
        return l, t, reg

    def is_nan(self, sess, feed_dict={}):
        return sess.run(tf.is_nan(self.tensor), feed_dict)