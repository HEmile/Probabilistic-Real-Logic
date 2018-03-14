from pascalpart import *
import tensorflow as tf
import random, os, pdb
import logictensornetworks as ltn
import config
import time

# swith between GPU and CPU
tf_config = tf.ConfigProto(device_count={'GPU': 1})

number_of_pairs_for_axioms = 1000

train_data, pairs_of_train_data, types_of_train_data, partOf_of_pairs_of_train_data, _, _ = get_data("train",
                                                                                                     max_rows=1000000000)

# computing positive and negative examples for types and partof

idxs_of_positive_examples_of_types = {}
idxs_of_negative_examples_of_types = {}

for type in selected_types:
    idxs_of_positive_examples_of_types[type] = np.where(types_of_train_data == type)[0]
    idxs_of_negative_examples_of_types[type] = np.where(types_of_train_data != type)[0]

idxs_of_positive_examples_of_partOf = np.where(partOf_of_pairs_of_train_data)[0]
idxs_of_negative_examples_of_partOf = np.where(partOf_of_pairs_of_train_data == False)[0]

existing_types = [t for t in selected_types if idxs_of_positive_examples_of_types[t].size > 0]

print("non empty types in train data", existing_types)
print("finished uploading and analyzing data")
print("Start model definition")

# Define the clauses in the knowledge base.
# First we define the facts.
clauses_for_positive_examples_of_types = \
    [ltn.Clause([ltn.Literal(True, isOfType[t], objects_of_type[t])], label="examples_of_" + t, weight=1.0) for t in
     existing_types]

clauses_for_negative_examples_of_types = \
    [ltn.Clause([ltn.Literal(False, isOfType[t], objects_of_type_not[t])], label="examples_of_not_" + t, weight=1.0) for
     t in existing_types]

clause_for_positive_examples_of_partOf = [ltn.Clause([ltn.Literal(True, isPartOf, object_pairs_in_partOf)],
                                                     label="examples_of_object_pairs_in_partof_relation", weight=1.0)]

clause_for_negative_examples_of_partOf = [ltn.Clause([ltn.Literal(False, isPartOf, object_pairs_not_in_partOf)],
                                                     label="examples_of_object_pairs_not_in_part_of_relation",
                                                     weight=1.0)]

# defining axioms from the partOf ontology
parts_of_whole, wholes_of_part = get_part_whole_ontology()

w1 = {}
p1 = {}
pw = {}
oo = ltn.Domain(number_of_features * 2 + 2, label="same_object_pairs")
o = ltn.Domain(number_of_features, label="a_generi_object")

w0 = ltn.Domain(number_of_features, label="whole_of_part_whole_pair")
p0 = ltn.Domain(number_of_features, label="part_of_part_whole_pair")
p0w0 = ltn.Domain(number_of_features * 2 + 2, label="part_whole_pair")
w0p0 = ltn.Domain(number_of_features * 2 + 2, label="whole_part_pair")

for t in selected_types:
    w1[t] = ltn.Domain(number_of_features, label="whole_predicted_objects_for_" + t)
    p1[t] = ltn.Domain(number_of_features, label="part_predicted_objects_for_" + t)
    pw[t] = ltn.Domain(number_of_features * 2 + 2, label="potential_part_whole_object_pairs_for_" + t)

partOf_is_antisymmetric = [ltn.Clause([ltn.Literal(False, isPartOf, p0w0), ltn.Literal(False, isPartOf, w0p0)],
                                      label="part_of_is_antisymmetric", weight=0.37)]

partof_is_irreflexive = [ltn.Clause([ltn.Literal(False, isPartOf, oo)],
                                    label="part_of_is_irreflexive", weight=0.37)]

clauses_for_parts_of_wholes = [ltn.Clause([ltn.Literal(False, isOfType[w], w1[w]),
                                           ltn.Literal(False, isPartOf, pw[w])] + \
                                          [ltn.Literal(True, isOfType[p], p1[w]) for p in parts_of_whole[w]],
                                          label="parts_of_" + w) for w in parts_of_whole.keys()]

clauses_for_wholes_of_parts = [ltn.Clause([ltn.Literal(False, isOfType[p], p1[p]),
                                           ltn.Literal(False, isPartOf, pw[p])] +
                                          [ltn.Literal(True, isOfType[w], w1[p]) for w in wholes_of_part[p]],
                                          label="wholes_of_" + p) for p in wholes_of_part.keys()]

clauses_for_disjoint_types = [ltn.Clause([ltn.Literal(False, isOfType[t], o),
                                          ltn.Literal(False, isOfType[t1], o)], label=t + "_is_not_" + t1) for t in
                              selected_types for t1 in selected_types if t < t1]

clause_for_at_least_one_type = [
    ltn.Clause([ltn.Literal(True, isOfType[t], o) for t in selected_types], label="an_object_has_at_least_one_type")]


# return partof_is_irreflexive + partOf_is_antisymmetric + clauses_for_wholes_of_parts + \
#     clauses_for_parts_of_wholes + clauses_for_disjoint_types + clause_for_at_least_one_type


def add_noise_to_data(noise_ratio):
    if noise_ratio > 0:
        freq_other = {}

        for t in selected_types:
            freq_other[t] = {}
            number_of_not_t = len(idxs_of_negative_examples_of_types[t])
            for t1 in selected_types:
                if t1 != t:
                    freq_other[t][t1] = np.float(len(idxs_of_positive_examples_of_types[t1])) / number_of_not_t

        noisy_data_idxs = np.random.choice(range(len(train_data)), int(len(train_data) * noise_ratio), replace=False)

        for idx in noisy_data_idxs:
            type_of_idx = types_of_train_data[idx]
            not_types_of_idx = np.setdiff1d(selected_types, type_of_idx)
            types_of_train_data[idx] = np.random.choice(not_types_of_idx,
                                                        p=np.array([freq_other[type_of_idx][t1] \
                                                                    for t1 in not_types_of_idx]))

        noisy_data_pairs_idxs = np.append(np.random.choice(np.where(partOf_of_pairs_of_train_data)[0], int(
            partOf_of_pairs_of_train_data.sum() * noise_ratio * 0.5)),
                                          np.random.choice(np.where(np.logical_not(partOf_of_pairs_of_train_data))[0],
                                                           int(
                                                               partOf_of_pairs_of_train_data.sum() * noise_ratio * 0.5)))

        for idx in noisy_data_pairs_idxs:
            partOf_of_pairs_of_train_data[idx] = not (partOf_of_pairs_of_train_data[idx])

    idxs_of_noisy_positive_examples_of_types = {}
    idxs_of_noisy_negative_examples_of_types = {}

    for type in selected_types:
        idxs_of_noisy_positive_examples_of_types[type] = np.where(types_of_train_data == type)[0]
        idxs_of_noisy_negative_examples_of_types[type] = np.where(types_of_train_data != type)[0]

    idxs_of_noisy_positive_examples_of_partOf = np.where(partOf_of_pairs_of_train_data)[0]
    idxs_of_noisy_negative_examples_of_partOf = np.where(partOf_of_pairs_of_train_data == False)[0]

    print("I have introduced the following errors")
    for t in selected_types:
        print("wrong positive", t, len(np.setdiff1d(idxs_of_noisy_positive_examples_of_types[t],
                                                    idxs_of_positive_examples_of_types[t])))
        print("wrong negative", t, len(np.setdiff1d(idxs_of_noisy_negative_examples_of_types[t],
                                                    idxs_of_negative_examples_of_types[t])))

    print("wrong positive partof", len(np.setdiff1d(idxs_of_noisy_positive_examples_of_partOf,
                                                    idxs_of_positive_examples_of_partOf)))
    print("wrong negative partof", len(np.setdiff1d(idxs_of_noisy_negative_examples_of_partOf,
                                                    idxs_of_negative_examples_of_partOf)))

    return idxs_of_noisy_positive_examples_of_types, idxs_of_noisy_negative_examples_of_types, \
           idxs_of_noisy_positive_examples_of_partOf, idxs_of_noisy_negative_examples_of_partOf


def train(KB_full, KB_facts, KB_rules, data, alg='nc', noise_ratio=0.0, data_ratio=config.RATIO_DATA[0],
          saturation_limit=0.90, lambda_2=config.LAMBDA_2):
    prior_mean = []
    prior_lambda = config.REGULARIZATION

    # defining the label of the background knowledge
    kb_label = "KB_" + alg + "_nr_" + str(noise_ratio) + "_dr_" + str(data_ratio)

    def train_fn(with_facts, with_constraints, iterations):
        train_kb = True
        for i in range(iterations):
            ti = time.time()
            if i % config.FREQ_OF_FEED_DICT_GENERATION == 0:
                train_kb = True
                feed_dict = get_feed_dict(data, pairs_of_train_data,
                                          with_constraints=with_constraints, with_facts=with_facts)
                feed_dict[KB.prior_mean] = prior_mean
                feed_dict[KB.prior_lambda] = prior_lambda
            if i + 1 % config.FREQ_OF_SAVE == 0:
                print('Saving the model to a file')
                KB.save(sess, kb_label)
            if train_kb:
                sat_level, normal_loss, reg_loss = KB.train(sess, feed_dict)
                if np.isnan(sat_level):
                    train_kb = False
                if normal_loss < 0:  # Using log-likelihood aggregation
                    sat_level = np.exp(-sat_level)
                if sat_level >= saturation_limit:
                    train_kb = False
            if i % config.FREQ_OF_PRINT == 0:
                print(i, 'Sat level', str(sat_level), 'loss', normal_loss, 'regularization', reg_loss, 'iteration time',
                      time.time() - ti)
        return feed_dict

    # Make sure the graph is cleaned up after each experiment run to reduce memory usage.
    if alg == 'prior':
        kb_label = "KB_" + alg + '_l2_' + str(lambda_2) + "_nr_" + str(noise_ratio) + "_dr_" + str(data_ratio)
        KB = KB_rules

        # start training
        init = tf.global_variables_initializer()
        with tf.Session(config=tf_config) as sess:
            sess.run(init)
            prior_mean = np.zeros(sess.run(KB.num_params, {}))

            feed_dict = train_fn(with_facts=False, with_constraints=True, iterations=config.MAX_PRIOR_TRAINING_IT)

            # Create the parameters for the informative prior
            prior_mean = sess.run(KB.omega, feed_dict)
            prior_lambda = lambda_2

    KB = KB_full if alg == 'wc' else KB_facts

    # start training
    init = tf.global_variables_initializer()
    sess = tf.Session(config=tf_config)
    sess.run(init)
    if len(prior_mean) == 0:
        prior_mean = np.zeros(sess.run(KB.num_params, {}))

    train_fn(with_facts=True, with_constraints=alg == 'wc', iterations=config.MAX_TRAINING_ITERATIONS)

    KB.save(sess, kb_label)
    print("end of training")
    sess.close()


def get_feed_dict(data, pairs_data, with_constraints=True, with_facts=True):
    # print("selecting new training data")

    idxs_of_pos_ex_of_types, idxs_of_neg_ex_of_types, \
    idxs_of_pos_ex_of_partOf, idxs_of_neg_ex_of_partOf = data

    feed_dict = {}

    if with_facts:
        # positive and negative examples for types
        for t in existing_types:
            feed_dict[objects_of_type[t].tensor] = \
                train_data[np.random.choice(idxs_of_pos_ex_of_types[t], config.N_POS_EXAMPLES_TYPES)][:, 1:]
            feed_dict[objects_of_type_not[t].tensor] = \
                train_data[np.random.choice(idxs_of_neg_ex_of_types[t], config.N_NEG_EXAMPLES_TYPES)][:, 1:]

        # positive and negative examples for partOF
        feed_dict[object_pairs_in_partOf.tensor] = \
            pairs_of_train_data[np.random.choice(idxs_of_pos_ex_of_partOf, config.N_POS_EXAMPLES_PARTOF)]

        feed_dict[object_pairs_not_in_partOf.tensor] = \
            pairs_of_train_data[np.random.choice(idxs_of_neg_ex_of_partOf, config.N_NEG_EXAMPLES_PARTOF)]
    # feed data for axioms
    tmp = pairs_data[np.random.choice(range(pairs_data.shape[0]), number_of_pairs_for_axioms)]
    feed_dict[o.tensor] = tmp[:, :number_of_features]

    if with_constraints:
        for t in selected_types:
            feed_dict[pw[t].tensor] = tmp
            feed_dict[w1[t].tensor] = tmp[:, number_of_features:2 * number_of_features]
            feed_dict[p1[t].tensor] = tmp[:, :number_of_features]

        feed_dict[oo.tensor] = np.concatenate([tmp[:, :number_of_features], tmp[:, :number_of_features],
                                               np.ones((tmp.shape[0], 2), dtype=float)], axis=1)
        feed_dict[p0w0.tensor] = tmp
        feed_dict[w0.tensor] = feed_dict[p0w0.tensor][:, number_of_features:2 * number_of_features]
        feed_dict[p0.tensor] = feed_dict[p0w0.tensor][:, :number_of_features]
        feed_dict[w0p0.tensor] = np.concatenate([
            feed_dict[w0.tensor], feed_dict[p0.tensor], feed_dict[p0w0.tensor][:, -1:-3:-1]], axis=1)
    # print("feed dict size is as follows")
    # for k in feed_dict:
    #     print(k.name, feed_dict[k].shape)
    return feed_dict


# defining the clauses of the background knowledge
facts = clauses_for_positive_examples_of_types + clauses_for_negative_examples_of_types + \
        clause_for_positive_examples_of_partOf + clause_for_negative_examples_of_partOf

rules = partof_is_irreflexive + partOf_is_antisymmetric + clauses_for_wholes_of_parts + \
        clauses_for_parts_of_wholes + clauses_for_disjoint_types + clause_for_at_least_one_type

# Lists all predicates
predicates = list(isOfType.values()) + [isPartOf]

# Create the different knowledge bases.
print('Defining knowledge bases')
KB_full = ltn.KnowledgeBase(predicates, facts + rules, "models/")
KB_facts = ltn.KnowledgeBase(predicates, facts, "models/")
KB_rules = ltn.KnowledgeBase(predicates, rules, "models/")

for nr in config.NOISE_VALUES:
    data = add_noise_to_data(nr)
    for alg in config.ALGORITHMS:
        lambda_2s = config.LAMBDA_2_VALUES if alg == 'prior' else [config.LAMBDA_2]
        for lambda_2 in lambda_2s:
            train(KB_full, KB_facts, KB_rules, data, alg=alg, noise_ratio=nr,
                  saturation_limit=config.SATURATION_LIMIT, lambda_2=lambda_2)
