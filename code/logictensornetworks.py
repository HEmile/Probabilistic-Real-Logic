import sys
import tensorflow as tf
import numpy as np
import pdb

default_layers = 10
default_smooth_factor = 0.0000001
default_tnorm = "product"
default_optimizer = "gd"
default_aggregator = "min"
default_positive_fact_penality = 0.0
default_clauses_aggregator = "min"
default_p_pmean = -3.0


def train_op(loss, optimization_algorithm):
    if optimization_algorithm == "ftrl":
        optimizer = tf.train.FtrlOptimizer(learning_rate=0.01, learning_rate_power=-0.5)
    if optimization_algorithm == "gd":
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
    if optimization_algorithm == "ada":
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
    if optimization_algorithm == "rmsprop":
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01, decay=0.9)
    return optimizer.minimize(loss)


def PR(tensor):
    global count
    np.set_printoptions(threshold=np.nan)
    return tf.Print(tensor, [tf.shape(tensor), tensor], summarize=200000)


def disjunction_of_literals(literals):
    list_of_literal_tensors = [lit.tensor for lit in literals]
    literals_tensor = tf.concat(list_of_literal_tensors, 1)
    if default_tnorm == "product":
        result = tf.ones([literals[0].rows, 1]) - tf.reduce_prod(tf.ones([literals[0].rows, 1]) - literals_tensor, 1,
                                                                 keep_dims=True)
    if default_tnorm == "yager2":
        result = tf.minimum(tf.ones([literals[0].rows, 1]),
                            tf.sqrt(tf.reduce_sum(tf.square(literals_tensor), 1, keep_dims=True)))
    if default_tnorm == "luk":
        result = tf.minimum(tf.ones([literals[0].rows, 1]), tf.reduce_sum(literals_tensor, 1, keep_dims=True))
    if default_tnorm == "goedel":
        result = tf.reduce_max(literals_tensor, 1)
    if default_aggregator == "product":
        return tf.reduce_prod(result, keep_dims=True)
    if default_aggregator == "mean":
        return tf.reduce_mean(result, keep_dims=True)
    if default_aggregator == "gmean":
        return tf.exp(tf.multiply(tf.reduce_sum(tf.log(result), keep_dims=True),
                             tf.inv(tf.to_float(tf.size(result)))))
    if default_aggregator == "pmean":
        default_p_pmean_tensor = tf.constant(default_p_pmean, shape=[literals.rows])
        return tf.pow(tf.multiply(tf.reduce_sum(tf.pow(result, default_p_pmean_tensor), keep_dims=True),
                             tf.inv(tf.to_float(tf.size(result)))), tf.inv(default_p_pmean))
    if default_aggregator == "hmean":
        return tf.multiply(tf.to_float(tf.size(result)), tf.inv(tf.reduce_sum(tf.inv(result), keep_dims=True)))
    if default_aggregator == "min":
        return tf.reduce_min(result, keep_dims=True)


def smooth(parameters):
    norm_of_omega = tf.reduce_sum(tf.expand_dims(tf.concat([tf.expand_dims(tf.reduce_sum(tf.square(par)), 0) for par in
                                                            parameters], 0), 1))
    return default_smooth_factor * norm_of_omega


def prod_of_domains(order, domains):
    cols_dom_ord0 = domains[order[0]].columns
    rows_doms_before_dom_ord0 = np.prod([domains[i].rows for i in range(order[0])], dtype=np.int32)
    rows_doms_after_dom_ord0 = np.prod([domains[i].rows for i in range(order[0] + 1, len(domains))], dtype=np.int32)
    rows_all_doms = np.prod([domains[i].rows for i in range(len(domains))], dtype=np.int32)
    d1 = tf.reshape(tf.tile(domains[order[0]].tensor, tf.constant([rows_doms_before_dom_ord0,
                                                                   rows_doms_after_dom_ord0])),
                    tf.constant([rows_all_doms, cols_dom_ord0]))
    if len(order) == 1:
        return d1
    else:
        return tf.concat([d1, prod_of_domains(order[1:], domains)], 1)


class Domain:
    def __init__(self, rows, columns, dom_type="float", label=None):
        self.rows = rows
        self.columns = columns
        self.label = label
        self.tensor = tf.placeholder(dom_type, shape=[self.rows, self.columns], name=self.label)
        self.parameters = []


class Domain_union(Domain):
    def __init__(self, domains):
        self.rows = np.sum([dom.rows for dom in domains])
        self.columns = domains[0].columns
        self.label = "union of" + ",".join([dom.label for dom in domains])
        self.tensor = tf.concat([dom.tensor for dom in domains], 0)
        self.parameters = [par for dom in domains for par in dom.parameters]


class Domain_concatenation(Domain):
    def __init__(self, domains):
        self.rows = domains[0].rows
        self.columns = np.sum([dom.columns for dom in domains])
        self.label = "concatenation of" + ",".join([dom.label for dom in domains])
        self.tensor = tf.concat([dom.tensor for dom in domains], 1)
        self.parameters = [par for dom in domains for par in dom.parameters]


class Constant(Domain):
    def __init__(self, label, value=None, domain=None):
        self.rows = 1
        self.label = label
        if value:
            self.tensor = tf.constant([value], dtype=tf.float32)
            self.parameters = []
            self.columns = len(value)
        else:
            self.columns = domain.columns
            self.C = tf.Variable(tf.random_normal([1, domain.rows], mean=1))
            self.tensor = tf.div(tf.matmul(tf.abs(self.C), domain.tensor),
                                 tf.reduce_sum(tf.abs(self.C)))
            self.parameters = [self.C]


class Function(Domain):
    def __init__(self, label, domains, ranges, value=None):
        self.label = label
        self.domains = domains
        self.ranges = ranges
        self.rows = np.prod([d.rows for d in domains])
        self.domain_columns = np.sum([d.columns for d in domains])
        self.columns = np.sum([d.columns for d in ranges])
        self.value = value
        if self.value:
            self.parameters = []
        else:
            self.M = tf.Variable(tf.random_normal([self.domain_columns,
                                                   self.columns]),
                                 name="M_" + self.label)

            self.n = tf.Variable(tf.random_normal([1, self.columns]),
                                 name="n_" + self.label)
            self.parameters = [self.n, self.M]
        if self.value:
            self.tensor = self.value
        else:
            standard_order = range(len(domains))
            X = prod_of_domains(standard_order, domains)
            multipleM = tf.tile(tf.expand_dims(self.M, 0), [self.rows, 1, 1])
            expandedX = tf.expand_dims(X, 1)
            self.tensor = tf.add(tf.squeeze(tf.matmul(expandedX, multipleM)), self.n)


class Predicate:
    def __init__(self, label, domains):
        self.label = label
        self.domains = domains
        self.rows = np.prod([d.rows for d in self.domains])
        self.columns = np.sum([d.columns for d in self.domains])
        self.number_of_layers = default_layers
        self.W = tf.Variable(tf.zeros([default_layers,
                                       self.columns,
                                       self.columns]),
                             name="W" + label)
        self.V = tf.Variable(tf.zeros([default_layers,
                                       self.columns]),
                             name="V" + label)
        self.b = tf.Variable(-(tf.ones([1, default_layers])),
                             name="b" + label)
        self.u = tf.Variable(tf.ones([default_layers, 1]),
                             name="u" + label)
        self.parameters = [self.W, self.V, self.b, self.u]

    def tensor(self, order=None, domains=None):
        if not domains:
            order = range(len(self.domains))
            domains = self.domains
        X_rows = np.prod([d.rows for d in domains])
        X_columns = self.columns
        X = prod_of_domains(order, domains)
        Wflat = tf.concat(tf.unstack(self.W), 1)
        XW = tf.matmul(X, Wflat)
        XWX = tf.squeeze(tf.matmul(tf.reshape(tf.expand_dims(XW, 1),
                                              [X_rows,
                                               self.number_of_layers,
                                               X_columns]),
                                   tf.expand_dims(X, 2)))
        VX = tf.matmul(X, tf.transpose(self.V))
        B = tf.tile(self.b, [X_rows, 1])
        gX = tf.matmul(tf.tanh(tf.add(XWX, tf.add(VX, B))), self.u)
        return tf.sigmoid(gX)


class Literal:
    def __init__(self, polarity, predicate, order=None, domains=None):
        self.predicate = predicate
        self.polarity = polarity
        if not order and not domains:
            self.order = [0]
            self.domains = predicate.domains
            self.rows = predicate.rows
            self.parameters = predicate.parameters
            if polarity:
                self.tensor = predicate.tensor()
            else:
                self.tensor = tf.ones([self.rows, 1]) - predicate.tensor()
        else:
            self.domains = domains
            self.order = order
            self.rows = np.prod([d.rows for d in self.domains])
            self.parameters = predicate.parameters + [par for par in [d.parameters for d in domains]]
            if polarity:
                self.tensor = self.predicate.tensor(order, domains)
            else:
                self.tensor = tf.ones([self.rows, 1]) - self.predicate.tensor(order, domains)


class Clause:
    def __init__(self, literals, label=None, weight=1.0):
        self.weight = weight
        self.label = label
        self.literals = literals
        self.rows = literals[0].rows
        self.tensor = disjunction_of_literals(self.literals)
        self.predicates = set([lit.predicate for lit in self.literals])
        self.parameters = [par for lit in literals for par in lit.parameters]


class KnowledgeBase:
    def __init__(self, label, clauses, save_path=""):
        print("defining the knowledge base", label)
        self.label = label
        self.clauses = clauses
        self.parameters = [par for cl in self.clauses for par in cl.parameters]
        if not self.clauses:
            self.tensor = tf.constant(1.0)
        else:
            clauses_value_tensor = tf.concat([cl.tensor for cl in clauses], 0)
            if default_clauses_aggregator == "min":
                self.tensor = tf.reduce_min(clauses_value_tensor)
            if default_clauses_aggregator == "wmean":
                weights_tensor = tf.constant([cl.weight for cl in clauses])
                self.tensor = PR(
                    tf.div(tf.reduce_sum(weights_tensor * clauses_value_tensor), tf.reduce_sum(weights_tensor)))
        self.loss = smooth(self.parameters) + \
                    tf.multiply(default_positive_fact_penality, self.penalize_positive_facts()) - self.tensor
        self.save_path = save_path
        self.train_op = train_op(self.loss, default_optimizer)
        self.saver = tf.train.Saver()

    def penalize_positive_facts(self):
        tensor_for_positive_facts = []
        for cl in self.clauses:
            for lit in cl.literals:
                tensor_for_positive_facts.append(
                    tf.reduce_sum(Literal(True, lit.predicate, lit.order, lit.domains).tensor, keep_dims=True))
        return tf.reduce_sum(tf.concat(tensor_for_positive_facts, 0))

    def save(self, sess, version=""):
        save_path = self.saver.save(sess, self.save_path + self.label + version + ".ckpt")

    def restore(self, sess):
        ckpt = tf.train.get_checkpoint_state(self.save_path)
        if ckpt and ckpt.model_checkpoint_path:
            print("restoring model")
            self.saver.restore(sess, ckpt.model_checkpoint_path)

    def train(self, sess, feed_dict={}):
        return sess.run(self.train_op, feed_dict)

    def is_nan(self, sess, feed_dict={}):
        return sess.run(tf.is_nan(self.tensor), feed_dict)
