from pascalpart import *
import tensorflow as tf
import random, os, pdb
import logictensornetworks as ltn
import config
import time
from evaluate import compute_confusion_matrix_pof, compute_measures, stat, adjust_prec, auc

# swith between GPU and CPU
tf_config = tf.ConfigProto(device_count={'GPU': 1})

# When using semi-supervised training, also load in the pairs for the complete dataset
if config.DO_SEMI_SUPERVISED:
    _, all_pairs_of_train_data, _, all_pairs_of_partOF_train_data, _, _ = get_data("train", max_rows=1000000000, data_ratio=1)
    idxs_of_all_positive_examples_of_partOf = np.where(all_pairs_of_partOF_train_data)[0]
    idxs_of_all_negative_examples_of_partOf = np.where(all_pairs_of_partOF_train_data == False)[0]

# loading test data
test_data, pairs_of_test_data, types_of_test_data, partOF_of_pairs_of_test_data, pairs_of_bb_idxs_test, pics = get_data(
    "test", max_rows=50000, data_ratio=1)
idxs_of_test_pos_examples_of_partOf = np.where(partOF_of_pairs_of_test_data)[0]

print("finished uploading and analyzing data")
print("Start model definition")

# Define the clauses in the knowledge base.
# First we define the facts.
clause_for_types = [ltn.TypeClause(mutExclType,
                                   weight=len(selected_types) * config.WEIGHT_TYPES_EXAMPLES, label='examples_types')]
labels_placeholder = clause_for_types[0].correct_labels

clause_for_positive_examples_of_partOf = [ltn.Clause([ltn.Literal(True, isPartOf, object_pairs_in_partOf)],
                                         label="examples_of_object_pairs_in_partof_relation", weight=config.WEIGHT_POS_PARTOF_EXAMPLES)]

clause_for_negative_examples_of_partOf = [ltn.Clause([ltn.Literal(False, isPartOf, object_pairs_not_in_partOf)],
                                                     label="examples_of_object_pairs_not_in_part_of_relation",
                                                     weight=config.WEIGHT_NEG_PARTOF_EXAMPLES)]

# defining axioms from the partOf ontology
parts_of_whole, wholes_of_part = get_part_whole_ontology()

w1 = {}
p1 = {}
pw = {}
oo = ltn.Domain(number_of_features * 2 + 2, label="same_object_pairs")

w0 = ltn.Domain(number_of_features, label="whole_of_part_whole_pair")
p0 = ltn.Domain(number_of_features, label="part_of_part_whole_pair")
p0w0 = ltn.Domain(number_of_features * 2 + 2, label="part_whole_pair")
w0p0 = ltn.Domain(number_of_features * 2 + 2, label="whole_part_pair")

for t in selected_types:
    w1[t] = ltn.Domain(number_of_features, label="whole_predicted_objects_for_" + t)
    p1[t] = ltn.Domain(number_of_features, label="part_predicted_objects_for_" + t)
    pw[t] = ltn.Domain(number_of_features * 2 + 2, label="potential_part_whole_object_pairs_for_" + t)
weight_ontology = tf.placeholder('float32', ())
if config.USE_IMPLICATION_CLAUSES:
    partOf_is_antisymmetric = [ltn.ImplicationClause(antecedent=[ltn.Literal(True, isPartOf, p0w0)],
                                                     consequent=[ltn.Literal(False, isPartOf, w0p0)],
                                          label="part_of_is_antisymmetric", weight=config.WEIGHT_LOGICAL_CLAUSES)]

    clauses_for_parts_of_wholes = [
        ltn.ImplicationClause(antecedent=[ltn.Literal(True, isOfType[w], w1[w], name=w),
                                          ltn.Literal(True, isPartOf, pw[w], name='partOf')],
                   consequent=[ltn.Literal(True, isOfType[p], p1[w], name=p) for p in parts_of_whole[w]],
                   label="parts_of_" + w, weight=weight_ontology) for w in parts_of_whole.keys()
        if len(parts_of_whole[w]) > 0]

    clauses_for_wholes_of_parts = [
        ltn.ImplicationClause(antecedent=[ltn.Literal(True, isOfType[p], p1[p], name=p),
                                          ltn.Literal(True, isPartOf, pw[p], name='partOf')],
                   consequent=[ltn.Literal(True, isOfType[w], w1[p], name=w) for w in wholes_of_part[p]],
                   label="wholes_of_" + p, weight=weight_ontology) for p in wholes_of_part.keys()
        if len(wholes_of_part[p]) > 0]
else:
    disable_prec_gradient = not config.CAN_ONTOLOGY_TRAIN_PRECEDENT

    partOf_is_antisymmetric = [ltn.Clause([ltn.Literal(False, isPartOf, p0w0, disable_gradient=disable_prec_gradient),
                                           ltn.Literal(False, isPartOf, w0p0)],
                                          label="part_of_is_antisymmetric", weight=config.WEIGHT_LOGICAL_CLAUSES)]

    clauses_for_parts_of_wholes = [ltn.Clause([ltn.Literal(False, isOfType[w], w1[w], disable_gradient=disable_prec_gradient, name=w),
                                               ltn.Literal(False, isPartOf, pw[w], disable_gradient=disable_prec_gradient, name='partOf')] + \
                                              [ltn.Literal(True, isOfType[p], p1[w], p, name=p) for p in parts_of_whole[w]],
                                              label="parts_of_" + w, weight=config.WEIGHT_ONTOLOGY_CLAUSES) for w in parts_of_whole.keys()]

    clauses_for_wholes_of_parts = [ltn.Clause([ltn.Literal(False, isOfType[p], p1[p], disable_gradient=disable_prec_gradient, name=p),
                                               ltn.Literal(False, isPartOf, pw[p], disable_gradient=disable_prec_gradient, name='partOf')] +
                                              [ltn.Literal(True, isOfType[w], w1[p], name=w) for w in wholes_of_part[p]],
                                              label="wholes_of_" + p, weight=config.WEIGHT_ONTOLOGY_CLAUSES) for p in wholes_of_part.keys() ]
partof_is_irreflexive = [ltn.Clause([ltn.Literal(False, isPartOf, oo)],
                                        label="part_of_is_irreflexive", weight=config.WEIGHT_LOGICAL_CLAUSES)]
# if not config.USE_MUTUAL_EXCL_PREDICATES:
# These are not needed when using the softmax output function. Also speeds up the computation massively
# not to have them :)
o = ltn.Domain(number_of_features, label="a_generi_object")

# clauses_for_disjoint_types = [ltn.Clause([ltn.Literal(False, isOfType[t], o, disable_gradient=disable_prec_gradient),
#                                           ltn.Literal(False, isOfType[t1], o)], label=t + "_is_not_" + t1) for t in
#                               selected_types for t1 in selected_types if t < t1]

# clause_for_at_least_one_type = [
#     ltn.Clause([ltn.Literal(True, isOfType[t], o) for t in selected_types],
#                label="an_object_has_at_least_one_type")]


# return partof_is_irreflexive + partOf_is_antisymmetric + clauses_for_wholes_of_parts + \
#     clauses_for_parts_of_wholes + clauses_for_disjoint_types + clause_for_at_least_one_type


# TODO: Update this to new data format of types
# def add_noise_to_data(noise_ratio):
#     random.seed(config.RANDOM_SEED)
#     if noise_ratio > 0:
#         freq_other = {}
#
#         for t in selected_types:
#             freq_other[t] = {}
#             number_of_not_t = len(idxs_of_negative_examples_of_types[t])
#             for t1 in selected_types:
#                 if t1 != t:
#                     freq_other[t][t1] = np.float(len(idxs_of_positive_examples_of_types[t1])) / number_of_not_t
#
#         noisy_data_idxs = np.random.choice(range(len(train_data)), int(len(train_data) * noise_ratio), replace=False)
#
#         for idx in noisy_data_idxs:
#             type_of_idx = types_of_train_data[idx]
#             not_types_of_idx = np.setdiff1d(selected_types, type_of_idx)
#             types_of_train_data[idx] = np.random.choice(not_types_of_idx,
#                                                         p=np.array([freq_other[type_of_idx][t1] \
#                                                                     for t1 in not_types_of_idx]))
#
#         noisy_data_pairs_idxs = np.append(np.random.choice(np.where(partOf_of_pairs_of_train_data)[0], int(
#             partOf_of_pairs_of_train_data.sum() * noise_ratio * 0.5)),
#                                           np.random.choice(np.where(np.logical_not(partOf_of_pairs_of_train_data))[0],
#                                                            int(
#                                                                partOf_of_pairs_of_train_data.sum() * noise_ratio * 0.5)))
#
#         for idx in noisy_data_pairs_idxs:
#             partOf_of_pairs_of_train_data[idx] = not (partOf_of_pairs_of_train_data[idx])
#
#     idxs_of_noisy_positive_examples_of_types = {}
#     idxs_of_noisy_negative_examples_of_types = {}
#
#     for type in selected_types:
#         idxs_of_noisy_positive_examples_of_types[type] = np.where(types_of_train_data == type)[0]
#         idxs_of_noisy_negative_examples_of_types[type] = np.where(types_of_train_data != type)[0]
#
#     idxs_of_noisy_positive_examples_of_partOf = np.where(partOf_of_pairs_of_train_data)[0]
#     idxs_of_noisy_negative_examples_of_partOf = np.where(partOf_of_pairs_of_train_data == False)[0]
#
#     print("I have introduced the following errors")
#     for t in selected_types:
#         print("wrong positive", t, len(np.setdiff1d(idxs_of_noisy_positive_examples_of_types[t],
#                                                     idxs_of_positive_examples_of_types[t])))
#         print("wrong negative", t, len(np.setdiff1d(idxs_of_noisy_negative_examples_of_types[t],
#                                                     idxs_of_negative_examples_of_types[t])))
#
#     print("wrong positive partof", len(np.setdiff1d(idxs_of_noisy_positive_examples_of_partOf,
#                                                     idxs_of_positive_examples_of_partOf)))
#     print("wrong negative partof", len(np.setdiff1d(idxs_of_noisy_negative_examples_of_partOf,
#                                                     idxs_of_negative_examples_of_partOf)))
#
#     return idxs_of_noisy_positive_examples_of_types, idxs_of_noisy_negative_examples_of_types, \
#            idxs_of_noisy_positive_examples_of_partOf, idxs_of_noisy_negative_examples_of_partOf


def train_fn(with_facts, with_constraints, iterations, KB, prior_mean, prior_lambda, sess, kb_label):
    random.seed(config.RANDOM_SEED)
    train_writer = tf.summary.FileWriter('logging/' + kb_label + '/train', graph=None)
    test_writer = tf.summary.FileWriter('logging/' + kb_label + '/test', graph=None)
    modus_ponens_writer = tf.summary.FileWriter('logging/' + kb_label + '/modus_ponens', graph=None)
    modus_tollens_writer = tf.summary.FileWriter('logging/' + kb_label + '/modus_tollens', graph=None)

    train_kb = True
    for i in range(iterations):
        ti = time.time()
        if i % config.FREQ_OF_FEED_DICT_GENERATION == 0:
            train_kb = True
            feed_dict = get_feed_dict(pairs_of_train_data, with_constraints=with_constraints, with_facts=with_facts)
            feed_dict[KB.prior_mean] = prior_mean
            feed_dict[KB.prior_lambda] = prior_lambda
            feed_dict[weight_ontology] = config.WEIGHT_ONTOLOGY_CLAUSES_START if \
                i < config.ITERATIONS_UNTIL_WEIGHT_SWAP else config.WEIGHT_ONTOLOGY_CLAUSES_END

        if train_kb:
            sat_level, normal_loss, reg_loss = KB.train(sess, feed_dict)
            if np.isnan(sat_level):
                train_kb = False
            if normal_loss < 0:  # Using log-likelihood aggregation
                sat_level = np.exp(-sat_level)
            if sat_level >= config.SATURATION_LIMIT:
                train_kb = False
        if i % config.FREQ_OF_PRINT == 0:
            iter_time = time.time() - ti
            feed_dict = get_feed_dict(pairs_of_train_data)
            for kb in [KB_facts, KB_rules, KB_full, KB_ontology, KB_logical]:
                feed_dict[kb.prior_mean] = prior_mean
                feed_dict[kb.prior_lambda] = prior_lambda
                feed_dict[weight_ontology] = config.WEIGHT_ONTOLOGY_CLAUSES_START if \
                    i < config.ITERATIONS_UNTIL_WEIGHT_SWAP else config.WEIGHT_ONTOLOGY_CLAUSES_END

            summary = sess.run(summary_merge, feed_dict)
            train_writer.add_summary(summary, i)

            print(i, 'Sat level', str(sat_level), 'loss', normal_loss, 'regularization', reg_loss, 'iteration time',
                  iter_time)
        if i + 1 % config.FREQ_OF_SAVE == 0:
            KB.save(sess, kb_label)
        if i % config.FREQ_OF_TEST == 0:
            predicted_types_values_tensor = tf.concat([isOfType[t].tensor() for t in selected_types], 1)
            predicted_partOf_value_tensor = ltn.Literal(True, isPartOf, pairs_of_objects).tensor

            feed_dict = {}
            feed_dict[objects.tensor] = test_data[:, 1:]
            feed_dict[pairs_of_objects.tensor] = pairs_of_test_data
            feed_dict[weight_ontology] = config.WEIGHT_ONTOLOGY_CLAUSES_START if \
                i < config.ITERATIONS_UNTIL_WEIGHT_SWAP else config.WEIGHT_ONTOLOGY_CLAUSES_END

            chosen_pairs = np.random.choice(range(pairs_of_test_data.shape[0]), config.NUMBER_PAIRS_AXIOMS_TESTING)
            # chosen_pairs = np.random.choice(idxs_of_test_pos_examples_of_partOf, config.NUMBER_PAIRS_AXIOMS_TESTING)

            feed_dict_rules(feed_dict, pairs_of_test_data[chosen_pairs])

            tensors_to_retrieve = [predicted_types_values_tensor, predicted_partOf_value_tensor, rules_summary] \
                                  + grad_MP_tensors + grad_MT_tensors + literal_debug1

            outputs = sess.run(tensors_to_retrieve, feed_dict)
            values_of_types = outputs[0]
            values_of_partOf = outputs[1]
            summary_r = outputs[2]
            r_ind = 3
            if config.USE_IMPLICATION_CLAUSES:
                grad_MP = outputs[3:3+len(rules_ontology)]
                r_ind = 3 + 2 * len(rules_ontology)
                grad_MT = outputs[3 + len(rules_ontology):r_ind]

            literals = outputs[r_ind:]

            is_pair_of = partOF_of_pairs_of_test_data[chosen_pairs]
            bb_ids = pairs_of_bb_idxs_test[chosen_pairs]

            n_correct_mp_reasoning = 0
            n_wrong_mp_reasoning = 0
            n_correct_mp_updates = 0
            n_pos_correct_mp_reasoning = 0

            tot_mp_grad_magn = 0
            correct_mp_reason_magn = 0
            correct_mp_upd_magn = 0

            n_correct_mt_reasoning = 0
            n_wrong_mt_reasoning = 0
            n_correct_mt_updates = 0
            n_pos_correct_mt_reasoning = 0

            tot_mt_grad_magn = 0
            correct_mt_reason_magn = 0
            correct_mt_upd_magn = 0

            tot_evals = len(rules_ontology) * len(chosen_pairs)

            for r_i in range(len(rules_ontology)):
                rule = rules_ontology[r_i]
                wholes_of = rule in clauses_for_wholes_of_parts
                if config.PRINT_GRAD_DEBUG:
                    print(' and '.join([l.name for l in rule.literals[:2]]) + "->" +
                          ' or '.join([l.name for l in rule.literals[2:]]))
                for j in range(len(chosen_pairs)):
                    label1, label2 = bb_ids[j][0], bb_ids[j][1]
                    t1 = types_of_test_data[label1]
                    t2 = types_of_test_data[label2]
                    x = t1 if wholes_of else t2
                    y = t2 if wholes_of else t1
                    conseq_true = y in [l.name for l in rule.literals[2:]]
                    ant_true = is_pair_of[j] and x == rule.literals[0].name
                    is_correct_mp_reasoning = ant_true and conseq_true
                    is_correct_mt_reasoning = (not ant_true) and (not conseq_true)

                    grad_mp_xy = 0
                    grad_mt_xy = 0

                    if is_correct_mp_reasoning:
                        n_pos_correct_mp_reasoning += 1
                    if is_correct_mt_reasoning:
                        n_pos_correct_mt_reasoning += 1
                    if config.USE_IMPLICATION_CLAUSES:
                        grad_mp_xy = grad_MP[r_i][j]
                        grad_mt_xy = grad_MT[r_i][j]
                    else:
                        # TODO: Update
                        gradients = []
                        for z in range(len(rule.literals)):
                            g = np.prod([1 - (0 if y == z else literals[r_i][j, y])
                                                      for y in range(len(rule.literals))])
                            gradients.append(g)
                            g = np.abs(g)
                            if g > 0.1:
                                if z < 2:
                                    tot_mt_grad_magn += g
                                else:
                                    tot_mp_grad_magn += g

                    tot_mp_grad_magn += grad_mp_xy
                    tot_mt_grad_magn += grad_mt_xy
                    if is_correct_mp_reasoning:
                        correct_mp_reason_magn += grad_mp_xy
                    if conseq_true:
                        correct_mp_upd_magn += grad_mp_xy

                    if is_correct_mt_reasoning:
                        correct_mt_reason_magn += grad_mt_xy
                    if not ant_true:
                        correct_mt_upd_magn += grad_mt_xy

                    if grad_mp_xy > 0.1:
                        if is_correct_mp_reasoning:
                            n_correct_mp_reasoning += 1
                        else:
                            n_wrong_mp_reasoning += 1
                        if conseq_true:
                            n_correct_mp_updates += 1
                    if grad_mt_xy > 0.1:
                        if is_correct_mt_reasoning:
                            n_correct_mt_reasoning += 1
                        else:
                            n_wrong_mt_reasoning += 1
                        if not ant_true:
                            n_correct_mt_updates += 1
                    if config.PRINT_GRAD_DEBUG:
                        print("Correct MP reason", is_correct_mp_reasoning, "Correct conseq update", conseq_true,
                              "Correct MT reason", is_correct_mt_reasoning, "Correct ant update", ant_true)
                        print(literals[r_i][j, :])
                        print(grad_mp_xy, grad_mt_xy)

            # Compute AUC of partof and precision of types
            cm = compute_confusion_matrix_pof(config.THRESHOLDS, values_of_partOf,
                                              pairs_of_test_data, partOF_of_pairs_of_test_data)
            measures = {}
            compute_measures(cm, 'test', measures)
            precision, recall = stat(measures, 'test')
            precision = adjust_prec(precision)
            auc_pof = auc(precision, recall)

            max_type_labels = np.argmax(values_of_types, 1)
            max_type_labels = selected_types[max_type_labels]
            correct = np.where(max_type_labels == types_of_test_data)[0]
            prec_types = len(correct) / len(max_type_labels)

            # Creating IR measures for the consequent updating
            n_conseq_gradient_updates = n_correct_mp_reasoning+n_wrong_mp_reasoning
            prec_conseq_gradient_update = recall_conseq_gradient_update = \
                f1_conseq_gradient_update = prec_conseq_update = 0
            if n_conseq_gradient_updates > 0:
                prec_conseq_gradient_update = n_correct_mp_reasoning/n_conseq_gradient_updates
                recall_conseq_gradient_update = n_correct_mp_reasoning / n_pos_correct_mp_reasoning
                prec_conseq_update = n_correct_mp_updates / n_conseq_gradient_updates
                f1_conseq_gradient_update = 0 if recall_conseq_gradient_update == 0 or prec_conseq_gradient_update == 0 \
                    else 2/(1/recall_conseq_gradient_update + 1/prec_conseq_gradient_update)

            avg_conseq_grad_magn = tot_mp_grad_magn / tot_evals
            ratio_magn_correct_mp_reason = correct_mp_reason_magn / tot_mp_grad_magn
            # Average gradient of possible correct inferences (should be high)
            recall_magn_correct_mp_reason = correct_mp_reason_magn / n_pos_correct_mp_reasoning
            ratio_magn_correct_mp_upd = correct_mp_upd_magn / tot_mp_grad_magn


            # Creating IR measures for the antecedent updating
            n_ant_gradient_updates = n_correct_mt_reasoning + n_wrong_mt_reasoning
            prec_ant_gradient_update = recall_ant_gradient_update = f1_ant_gradient_update = prec_ant_update = 0
            if n_ant_gradient_updates > 0:
                prec_ant_gradient_update = n_correct_mt_reasoning / n_ant_gradient_updates
                recall_ant_gradient_update = n_correct_mt_reasoning / n_pos_correct_mt_reasoning
                prec_ant_update = n_correct_mt_updates / n_ant_gradient_updates
                f1_ant_gradient_update = 0 if recall_ant_gradient_update == 0 or prec_ant_gradient_update == 0 \
                    else 2 / (1 / recall_ant_gradient_update + 1 / prec_ant_gradient_update)

            avg_ant_grad_magn = tot_mt_grad_magn / tot_evals
            ratio_magn_correct_mt_reason = correct_mt_reason_magn / tot_mt_grad_magn
            # Average gradient of possible correct inferences (should be high)
            recall_magn_correct_mt_reason = correct_mt_reason_magn / n_pos_correct_mt_reasoning
            ratio_magn_correct_mt_upd = correct_mt_upd_magn / tot_mt_grad_magn

            tot_grad_magn = tot_mp_grad_magn + tot_mt_grad_magn
            fract_conseq_gradient = 0 if tot_grad_magn == 0 else \
                tot_mp_grad_magn / tot_grad_magn
            fract_ant_gradient = 0 if tot_grad_magn == 0 else \
                tot_mt_grad_magn / tot_grad_magn

            summary_t = tf.Summary(value=[
                tf.Summary.Value(tag="test/auc_pof", simple_value=auc_pof),
                tf.Summary.Value(tag="test/prec_types", simple_value=prec_types),
            ])

            summary_mp = tf.Summary(value=[
                tf.Summary.Value(tag="gradients/prec_reasoning", simple_value=prec_conseq_gradient_update),
                tf.Summary.Value(tag="gradients/recall_reasoning", simple_value=recall_conseq_gradient_update),
                tf.Summary.Value(tag="gradients/f1_reasoning", simple_value=f1_conseq_gradient_update),
                tf.Summary.Value(tag="gradients/num_update", simple_value=n_conseq_gradient_updates),
                tf.Summary.Value(tag="gradients/prec_update", simple_value=prec_conseq_update),

                tf.Summary.Value(tag="gradients/avg_magnitude_update", simple_value=avg_conseq_grad_magn),
                tf.Summary.Value(tag="gradients/ratio_of_gradient", simple_value=fract_conseq_gradient),
                tf.Summary.Value(tag="gradients/ratio_magn_correct_reason", simple_value=ratio_magn_correct_mp_reason),
                tf.Summary.Value(tag="gradients/avg_magn_correct_reason", simple_value=recall_magn_correct_mp_reason),
                tf.Summary.Value(tag="gradients/ratio_magn_correct_update", simple_value=ratio_magn_correct_mp_upd)
            ])

            summary_mt = tf.Summary(value=[
                tf.Summary.Value(tag="gradients/prec_reasoning", simple_value=prec_ant_gradient_update),
                tf.Summary.Value(tag="gradients/recall_reasoning", simple_value=recall_ant_gradient_update),
                tf.Summary.Value(tag="gradients/f1_reasoning", simple_value=f1_ant_gradient_update),
                tf.Summary.Value(tag="gradients/num_update", simple_value=n_ant_gradient_updates),
                tf.Summary.Value(tag="gradients/prec_update", simple_value=prec_ant_update),

                tf.Summary.Value(tag="gradients/avg_magnitude_update", simple_value=avg_ant_grad_magn),
                tf.Summary.Value(tag="gradients/ratio_of_gradient", simple_value=fract_ant_gradient),
                tf.Summary.Value(tag="gradients/ratio_magn_correct_reason", simple_value=ratio_magn_correct_mt_reason),
                tf.Summary.Value(tag="gradients/avg_magn_correct_reason", simple_value=recall_magn_correct_mt_reason),
                tf.Summary.Value(tag="gradients/ratio_magn_correct_update", simple_value=ratio_magn_correct_mt_upd)
            ])

            test_writer.add_summary(summary_r, i)
            test_writer.add_summary(summary_t, i)

            modus_ponens_writer.add_summary(summary_mp, i)
            modus_tollens_writer.add_summary(summary_mt, i)

    train_writer.flush()
    test_writer.flush()
    modus_ponens_writer.flush()
    modus_tollens_writer.flush()

    return feed_dict, auc_pof, prec_types


def train(KB, seed, alg='nc', noise_ratio=0.0, data_ratio=config.RATIO_DATA[0],
          lambda_2=config.LAMBDA_2, prior_mean=None):
    prior_lambda = config.REGULARIZATION

    alg_label = ("_SEMI_" if config.DO_SEMI_SUPERVISED else "") + "_nr_" + str(noise_ratio) + "_dr_" + str(data_ratio) \
                + config.DATASET + 'TNORM' + config.TNORM + 'SNORM' + config.SNORM + 'FORALL' + config.FORALL_AGGREGATOR + \
                'CLAUSE' + config.CLAUSE_AGGREGATOR + '_' + config.EXPERIMENT_NAME + '_SEED_' + str(seed)

    # defining the label of the background knowledge
    kb_label = "KB_" + alg + alg_label

    if alg == 'prior':
        kb_label = "KB_" + alg + '_l2_' + str(lambda_2) + alg_label
        prior_lambda = lambda_2

    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        # start training
        if prior_mean is None:
            prior_mean = np.zeros(sess.run(KB.num_params, {}))
        else:
            saver = tf.train.Saver()
            saver.restore(sess, 'models/prior.ckpt')

        _, auc_pof, prec_types = train_fn(with_facts=True, with_constraints=alg in ['wc', 'wconto', 'wclogic'],
              iterations=config.MAX_TRAINING_ITERATIONS, KB=KB, prior_mean=prior_mean, prior_lambda=prior_lambda,
                 sess=sess, kb_label=kb_label)

        KB.save(sess, kb_label)
        print("end of training")
    return auc_pof, prec_types


def feed_dict_rules(feed_dict, pairs_data, with_constraints=True):
    # feed data for axioms
    tmp = pairs_data
    # if not config.USE_MUTUAL_EXCL_PREDICATES:
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
    return feed_dict


def get_feed_dict(pairs_data, with_constraints=True, with_facts=True):
    feed_dict = {}

    if with_facts:
        chosen_examples = np.random.choice(np.size(train_data, 0), config.N_EXAMPLES_TYPES)
        feed_dict[objects.tensor] = train_data[chosen_examples][:, 1:]
        feed_dict[labels_placeholder] = [labelOfType[t] for t in types_of_train_data[chosen_examples]]

        # positive and negative examples for partOF
        if not config.CHEAT_SEMI_SUPERVISED:
            feed_dict[object_pairs_in_partOf.tensor] = \
                pairs_of_train_data[np.random.choice(idxs_of_positive_examples_of_partOf, config.N_POS_EXAMPLES_PARTOF)]

            feed_dict[object_pairs_not_in_partOf.tensor] = \
                pairs_of_train_data[np.random.choice(idxs_of_negative_examples_of_partOf, config.N_NEG_EXAMPLES_PARTOF)]
        else:
            feed_dict[object_pairs_in_partOf.tensor] = \
                all_pairs_of_train_data[np.random.choice(idxs_of_all_positive_examples_of_partOf, config.N_POS_EXAMPLES_PARTOF)]

            feed_dict[object_pairs_not_in_partOf.tensor] = \
                all_pairs_of_train_data[np.random.choice(idxs_of_all_negative_examples_of_partOf, config.N_NEG_EXAMPLES_PARTOF)]

    if config.DO_SEMI_SUPERVISED:
        if config.CHEAT_SEMI_SUPERVISED:
            chosen_pairs = np.random.choice(idxs_of_all_positive_examples_of_partOf, config.number_of_pairs_for_axioms)
        else:
            chosen_pairs = np.random.choice(range(all_pairs_of_train_data.shape[0]), config.number_of_pairs_for_axioms)
        feed_dict_rules(feed_dict, all_pairs_of_train_data[chosen_pairs], with_constraints)
    else:
        feed_dict_rules(feed_dict,
                        pairs_data[np.random.choice(range(pairs_data.shape[0]), config.number_of_pairs_for_axioms)],
                        with_constraints)
    return feed_dict


# defining the clauses of the background knowledge
facts = clause_for_types + clause_for_positive_examples_of_partOf + clause_for_negative_examples_of_partOf

rules_ontology = clauses_for_wholes_of_parts + clauses_for_parts_of_wholes
rules_logical = partof_is_irreflexive + partOf_is_antisymmetric
rules = rules_ontology + rules_logical

# if not config.USE_MUTUAL_EXCL_PREDICATES:
# rules += clauses_for_disjoint_types + clause_for_at_least_one_type

for rule in rules:
    tensor = rule.tensor
    tf.summary.scalar('rules/' + rule.label, tensor, collections=['rules'])

# for fact in facts:
#     tensor = fact.tensor
#     tf.summary.scalar('facts/' + fact.label, tensor, collections=['rules'])

clause_to_debug = clauses_for_parts_of_wholes[0]
for literal in clause_to_debug.literals:
    tensor = tf.reshape(tf.reduce_mean(literal.tensor), ())
    tf.summary.scalar('rules/' + clause_to_debug.label + '/' + literal.predicate.label + '_' +
                      literal.domain.label + str(literal.polarity), tensor, collections=['rules'])

clause_to_debug = clauses_for_parts_of_wholes[1]
for literal in clause_to_debug.literals:
    tensor = tf.reshape(tf.reduce_mean(literal.tensor), ())
    tf.summary.scalar('rules/' + clause_to_debug.label + '/' + literal.predicate.label + '_' +
                      literal.domain.label + str(literal.polarity), tensor, collections=['rules'])

clause_to_debug = clauses_for_parts_of_wholes[0]
literal_debug1 = [tf.concat([literal.tensor for literal in rule.literals], axis=1)
    for rule in rules_ontology]

grad_MP_tensors = []
grad_MT_tensors = []
if config.USE_IMPLICATION_CLAUSES:
    grad_MP_tensors = [rule.grad_mp for rule in rules_ontology]
    grad_MT_tensors = [rule.grad_mt for rule in rules_ontology]

# Lists all predicates
predicates = list(isOfType.values()) + [isPartOf]
mutExPredicates = [mutExclType] if config.USE_MUTUAL_EXCL_PREDICATES else []

# Create the different knowledge bases.
print('Defining knowledge bases')
KB_full = ltn.KnowledgeBase(predicates, mutExPredicates, facts + rules, "full", "models/")
KB_facts = ltn.KnowledgeBase(predicates, mutExPredicates, facts, "facts", "models/")
KB_rules = ltn.KnowledgeBase(predicates, mutExPredicates, rules, "rules", "models/")

KB_ontology = ltn.KnowledgeBase(predicates, mutExPredicates, rules_ontology, "ontology", "models/")
KB_logical = ltn.KnowledgeBase(predicates, mutExPredicates, rules_logical, "logical", "models/")

# The summary contains the loss tensors for the three Knowledge Bases defined up here.
summary_merge = tf.summary.merge_all(key='train')
rules_summary = tf.summary.merge_all(key='rules')

for nr in config.NOISE_VALUES:
    results = []

    with open('results' + config.EXPERIMENT_NAME + '.csv', 'w') as csvfile:
        csvfile.writelines('seed,' + ','.join([alg + 'auc_pof,' + alg + 'prec_types' for alg in config.ALGORITHMS]) + '\n')
        for eval in range(config.AMOUNT_OF_EVALUATIONS):
            results.append([])
            print('Starting eval', eval)

            seed = config.RANDOM_SEED + eval

            # Loading training data
            train_data, pairs_of_train_data, types_of_train_data, \
            partOf_of_pairs_of_train_data, _, _ = get_data("train", max_rows=1000000000, seed=seed)

            # computing positive and negative examples for types and partof
            idxs_of_positive_examples_of_partOf = np.where(partOf_of_pairs_of_train_data)[0]
            idxs_of_negative_examples_of_partOf = np.where(partOf_of_pairs_of_train_data == False)[0]

            # Compute the prior mean for the current dataset
            if 'prior' in config.ALGORITHMS:
                with tf.Session(config=tf_config) as sess:
                    sess.run(tf.global_variables_initializer())
                    prior_mean = np.zeros(sess.run(KB_rules.num_params, {}))

                    feed_dict, _, _ = train_fn(with_facts=False, with_constraints=True, iterations=config.MAX_PRIOR_TRAINING_IT,
                                               KB=KB_rules, prior_mean=prior_mean, prior_lambda=config.REGULARIZATION,
                                               sess=sess, kb_label='prior_training')

                    KB_rules.save(sess, 'prior')

                    # Create the parameters for the informative prior
                    prior_mean = sess.run(KB_rules.omega, feed_dict)

            for alg in config.ALGORITHMS:
                lambda_2s = config.LAMBDA_2_VALUES if alg == 'prior' else [config.LAMBDA_2]
                for lambda_2 in lambda_2s:
                    if alg == 'wconto':
                        auc_pof, prec_types = train(KB_ontology, seed, alg=alg, noise_ratio=nr, prior_mean=None)
                    elif alg == 'wclogic':
                        auc_pof, prec_types = train(KB_logical, seed, alg=alg, noise_ratio=nr, prior_mean=None)
                    else:
                        auc_pof, prec_types = train(KB_full if alg == 'wc' else KB_facts, seed, alg=alg, noise_ratio=nr,
                                                    lambda_2=lambda_2, prior_mean=prior_mean if alg == 'prior' else None)
                    print('EVAL FINISHED')
                    print('eval', eval, 'AUC POF', auc_pof, 'prec types', prec_types)
                    results[eval].append(auc_pof)
                    results[eval].append(prec_types)

            csvfile.writelines(str(seed) + ',' + ','.join([str(r) for r in results[eval]]) + '\n')
        print('FINISHED THE COMPLETE RUN')
        print('Results:')
        for i in range(config.AMOUNT_OF_EVALUATIONS):
            print('Evaluation', i, ':', results[i])
