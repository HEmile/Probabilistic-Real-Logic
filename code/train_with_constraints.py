#!/usr/bin/env python

from pascalpart import *
import tensorflow as tf
import random, os, pdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# swith between GPU and CPU
config = tf.ConfigProto(device_count={'GPU': 1})

ltn.default_optimizer = "rmsprop"
model_path="training_with_constraints/"

number_of_positive_examples_x_types=100
number_of_positive_examples_x_partof=100
number_of_negative_examples_x_partof=6000
number_of_object_pairs_x_partof_axioms=10000

train_data = get_train_data()
len_of_train_data = len(train_data)
test_data = get_test_data()
type_of_train_data = get_types_of_train_data()
idx_of_whole_for_train_data = get_partof_of_train_data(train_data)

train_pics = get_pics(train_data)
predicted_pics = get_pics(test_data)

existing_types = []
for t in types:
    if t in type_of_train_data:
        existing_types.append(t)

threshold_side = 6.0/500

idxs_of_positive_examples_of_partof = [[idx, idx_of_whole_for_train_data[idx]]
                                       for idx in range(len_of_train_data)
                                       if idx_of_whole_for_train_data[idx] >= 0 and
                                       (train_data[idx][63] - train_data[idx][61] > threshold_side) and
                                       (train_data[idx][64] - train_data[idx][62] > threshold_side) and
                                       (train_data[idx_of_whole_for_train_data[idx]][63] - train_data[idx_of_whole_for_train_data[idx]][61] > threshold_side) and
                                       (train_data[idx_of_whole_for_train_data[idx]][64] - train_data[idx_of_whole_for_train_data[idx]][62] > threshold_side)]

idxs_of_negative_examples_of_partof = []

for idx_pic in train_pics:
    for idx_part in train_pics[idx_pic]:
        for idx_whole in train_pics[idx_pic]:
            if idx_of_whole_for_train_data[idx_part] != idx_whole and \
                    (train_data[idx_part][63] - train_data[idx_part][61] > threshold_side) and\
                    (train_data[idx_part][64] - train_data[idx_part][62] > threshold_side) and \
                    (train_data[idx_whole][63] - train_data[idx_whole][61] > threshold_side) and \
                    (train_data[idx_whole][64] - train_data[idx_whole][62] > threshold_side):
                idxs_of_negative_examples_of_partof.append([idx_part,idx_whole])

idxs_of_positive_examples_of_types = {}
idxs_of_negative_examples_of_types = {}

for t in existing_types:
    idxs_of_positive_examples_of_types[t] = np.array([idx for idx in range(len(type_of_train_data)) if type_of_train_data[idx] == t and
                                                      (train_data[idx][63] - train_data[idx][61] > threshold_side) and
                                                      (train_data[idx][64] - train_data[idx][62] > threshold_side)])
for t in types:
    idxs_of_negative_examples_of_types[t] = np.array([idx for idx in range(len(type_of_train_data)) if type_of_train_data[idx] != t and
                                                      (train_data[idx][63] - train_data[idx][61] > threshold_side) and
                                                      (train_data[idx][64] - train_data[idx][62] > threshold_side)])

# domain definition
objects_of_type = {}
objects_of_type_not = {}
for t in types:
    objects_of_type[t] = ltn.Domain(number_of_positive_examples_x_types,
                                    number_of_features,
                                    label="objects_of_type_"+t)
    objects_of_type_not[t] = ltn.Domain(
        number_of_positive_examples_x_types * (len(types) - 2),
        number_of_features,
        label="objects_of_type_not_" + t)
    objects_of_type_not["background"] = ltn.Domain(
        number_of_positive_examples_x_types * (len(types) - 1),
        number_of_features,
        label="objects_of_type_not_background")

object_pairs_in_partOf_relation = ltn.Domain(number_of_positive_examples_x_partof,
                                                 number_of_features * 2 + 2,
                                                 label="object_pairs_in_partof_relation")
object_pairs_not_in_partOf_relation = ltn.Domain(number_of_negative_examples_x_partof,
                                                 number_of_features * 2 + 2,
                                                 label="object_pairs_not_in_partof_relation")

clauses_for_positive_examples_of_types = \
    [ltn.Clause([ltn.Literal(True,isOfType[t],[0],[objects_of_type[t]])],
                       label="examples_of_"+t,weight=1.0) for t in existing_types]

clauses_for_negative_examples_of_types = \
    [ltn.Clause([ltn.Literal(False,isOfType[t],[0],[objects_of_type_not[t]])],
                       label="examples_of_not_"+t,weight=1.0) for t in types]

clause_for_positive_examples_of_partof = [ltn.Clause([ltn.Literal(True,isPartOf,[0],
                                                                  [object_pairs_in_partOf_relation])],
                                                      label="examples_of_object_pairs_in_partof_relation",weight=1.0)]

clause_for_negative_examples_of_partof = [ltn.Clause([ltn.Literal(False, isPartOf, [0],
                                                                  [object_pairs_not_in_partOf_relation])],
                                                     label="examples_of_object_pairs_not_in_part_of_relation",weight=1.0)]

idxs_of_part_whole_object_pairs_in_predicted_pics = [[idx_part, idx_whole]
                                                     for p in predicted_pics
                                                     for idx_part in predicted_pics[p]
                                                     for idx_whole in predicted_pics[p] if
                                                     (test_data[idx_part][63] - test_data[idx_part][61] > threshold_side) and
                                                     (test_data[idx_part][64] - test_data[idx_part][62] > threshold_side) and
                                                     (test_data[idx_whole][63] - test_data[idx_whole][61] > threshold_side) and
                                                     (test_data[idx_whole][64] - test_data[idx_whole][62] > threshold_side)]
                                                    
potential_whole_object = ltn.Domain(number_of_object_pairs_x_partof_axioms, number_of_features, label="whole_predicted_objects")
potential_part_object = ltn.Domain(number_of_object_pairs_x_partof_axioms, number_of_features, label="part_predicted_objects")
potential_part_whole_object_pair = ltn.Domain(number_of_object_pairs_x_partof_axioms, number_of_features*2 +2, label="potential_part_whole_object_pairs")
potential_whole_part_object_pair = ltn.Domain(number_of_object_pairs_x_partof_axioms, number_of_features*2 +2, label="potential_whole_part_object_pairs")
potential_same_object_pair = ltn.Domain(number_of_object_pairs_x_partof_axioms, number_of_features*2 +2, label="same_object_pairs")

clauses_for_partof_ontology = []
parts, wholes = get_part_whole_ontology()

clauses_for_partof_ontology.append(ltn.Clause([ltn.Literal(False,isPartOf,[0],[potential_part_whole_object_pair]),
                                           ltn.Literal(False,isPartOf,[0],[potential_whole_part_object_pair])],
                                          label="part_of_is_antisymmetric",weight=1.0))

clauses_for_partof_ontology.append(ltn.Clause([ltn.Literal(False, isPartOf, [0], [potential_same_object_pair])],label="part_of_is_irreflexive",weight=1.0))

for t in parts:
    clauses_for_partof_ontology.append(
        ltn.Clause(
            [ltn.Literal(False, isOfType[t], [0], [potential_whole_object])] +
            [ltn.Literal(False, isPartOf, [0], [potential_part_whole_object_pair])] +
            [ltn.Literal(True, isOfType[t1], [0], [potential_part_object]) for t1 in parts[t]],
            label="parts_of_"+t,weight=1.0))
    clauses_for_partof_ontology.append(
        ltn.Clause(
            [ltn.Literal(False, isOfType[t], [0], [potential_whole_object])] +
            [ltn.Literal(False, isPartOf, [0], [potential_whole_part_object_pair])],
            label=t+"_is_not_part_of_anything",weight=0.25))
for t in wholes:
    clauses_for_partof_ontology.append(
        ltn.Clause(
            [ltn.Literal(False, isOfType[t], [0], [potential_part_object])] +
            [ltn.Literal(False, isPartOf, [0], [potential_part_whole_object_pair])] +
            [ltn.Literal(True, isOfType[t1], [0], [potential_whole_object]) for t1 in wholes[t]],
            label="wholes_of_" + t,weight=1.0))
    clauses_for_partof_ontology.append(
        ltn.Clause(
            [ltn.Literal(False, isOfType[t], [0], [potential_part_object])] +
            [ltn.Literal(False, isPartOf, [0], [potential_whole_part_object_pair])],
            label=t+"_has_no_parts",weight=0.25))
"""
# constraints for sofa, chair, boat diningtable
for t in [types[5], types[27], types[30], types[38]]:
    clauses_for_partof_ontology.append(
        ltn.Clause(
            [ltn.Literal(False, isOfType[t], [0], [potential_whole_object])] +
            [ltn.Literal(False, isPartOf, [0], [potential_whole_part_object_pair])],
            label=t + "_is_not_part_of_anything", weight=0.29))

    clauses_for_partof_ontology.append(
        ltn.Clause(
            [ltn.Literal(False, isOfType[t], [0], [potential_whole_object])] +
            [ltn.Literal(False, isPartOf, [0], [potential_part_whole_object_pair])],
            label=t + "_has_no_parts", weight=0.29))
"""
KB = ltn.KnowledgeBase("KB_for_pascalpart",
                       clauses_for_positive_examples_of_types +
                       clauses_for_negative_examples_of_types +
                       clause_for_positive_examples_of_partof +
                       clause_for_negative_examples_of_partof +
                       clauses_for_partof_ontology,
                       model_path)

def get_feed_dict():
    global idxs_of_positive_examples_of_types, \
        idxs_of_positive_examples_of_partof, \
        idxs_of_negative_examples_of_partof, \
        test_data
    feed_dict = {}
    for t in existing_types:
        feed_dict[objects_of_type[t].tensor] = [train_data[idx][1:] for idx in [random.choice(idxs_of_positive_examples_of_types[t]) for _ in range(number_of_positive_examples_x_types)]]
    for t in types:
        feed_dict[objects_of_type_not[t].tensor] = sum([feed_dict[objects_of_type[t1].tensor] for t1 in existing_types if t1 != t],[])

    feed_dict[object_pairs_in_partOf_relation.tensor] = [
            np.concatenate([train_data[idx[0]][1:], train_data[idx[1]][1:], compute_extra_features(train_data[idx[0]][-4:]*500, train_data[idx[1]][-4:]*500)]) \
            for idx in [random.choice(idxs_of_positive_examples_of_partof) for _ in range(number_of_positive_examples_x_partof)]]

    feed_dict[object_pairs_not_in_partOf_relation.tensor] = [
        np.concatenate([train_data[idx[0]][1:],train_data[idx[1]][1:], compute_extra_features(train_data[idx[0]][-4:]*500, train_data[idx[1]][-4:]*500)])
        for idx in [random.choice(idxs_of_negative_examples_of_partof) for _ in range(number_of_negative_examples_x_partof)]]

    balanceParameter = 0.5

    random_idx_pairs_for_potential_part_whole_pairs = [random.choice(idxs_of_part_whole_object_pairs_in_predicted_pics)
                                                       for _ in range(int(balanceParameter * number_of_object_pairs_x_partof_axioms))]

    random_idx_pairs_for_positive_part_whole_pairs = [random.choice(idxs_of_positive_examples_of_partof)
                                                      for _ in range(int((1 - balanceParameter) * number_of_object_pairs_x_partof_axioms))]

    feed_dict[potential_part_object.tensor] = [test_data[idx_pair[0]][1:] \
                                               for idx_pair in random_idx_pairs_for_potential_part_whole_pairs] + \
                                              [train_data[idx_pair[0]][1:] for idx_pair in
                                               random_idx_pairs_for_positive_part_whole_pairs]

    feed_dict[potential_whole_object.tensor] = [test_data[idx_pair[1]][1:] \
                                                for idx_pair in random_idx_pairs_for_potential_part_whole_pairs] + \
                                               [train_data[idx_pair[1]][1:] for idx_pair in
                                                random_idx_pairs_for_positive_part_whole_pairs]

    feed_dict[potential_part_whole_object_pair.tensor] = [np.concatenate([test_data[idx_pair[0]][1:], test_data[idx_pair[1]][1:], compute_extra_features(test_data[idx_pair[0]][-4:]*500, test_data[idx_pair[1]][-4:]*500)]) \
                                                          for idx_pair in random_idx_pairs_for_potential_part_whole_pairs] + \
                                                         [np.concatenate([train_data[idx_pair[0]][1:], train_data[idx_pair[1]][1:], compute_extra_features(train_data[idx_pair[0]][-4:]*500, train_data[idx_pair[1]][-4:]*500)])
                                                          for idx_pair in random_idx_pairs_for_positive_part_whole_pairs]

    feed_dict[potential_whole_part_object_pair.tensor] = [np.concatenate([test_data[idx_pair[1]][1:], test_data[idx_pair[0]][1:], compute_extra_features(test_data[idx_pair[1]][-4:]*500, test_data[idx_pair[0]][-4:]*500)]) \
                                                          for idx_pair in random_idx_pairs_for_potential_part_whole_pairs] + \
                                                         [np.concatenate([train_data[idx_pair[1]][1:], train_data[idx_pair[0]][1:], compute_extra_features(train_data[idx_pair[1]][-4:]*500, train_data[idx_pair[0]][-4:]*500)])
                                                          for idx_pair in random_idx_pairs_for_positive_part_whole_pairs]

    feed_dict[potential_same_object_pair.tensor] = [
        np.concatenate([test_data[idx][1:], test_data[idx][1:], compute_extra_features(test_data[idx][-4:]*500, test_data[idx][-4:]*500)]) for idx in
        [random.choice(range(len(test_data))) for _ in range(number_of_object_pairs_x_partof_axioms)]]

    return feed_dict

init = tf.initialize_all_variables()
sess = tf.Session(config=config)
sess.run(init) # comment if restore previous model
#KB.restore(sess)
f = open(model_path + 'KB_satisfiability_evolution.txt','w')
iterations=[]
KB_satiasfiability=[]
oldval = 0.0

for i in range(8000):

    if i%50 == 0:
        feed_dict = get_feed_dict()
    KB.train(sess, feed_dict)

    if i%10 == 0:
        val = sess.run(KB.tensor,feed_dict)
        print i,"----->",val
        iterations.append(i)
        KB_satiasfiability.append(val)
        if val > oldval:
            KB.save(sess)
            oldval = val
    else:
        print i

plt.plot(iterations, KB_satiasfiability, 'ro')
plt.savefig(model_path + 'KB_satisfiability_evolution.png')
np.savetxt(f, KB_satiasfiability)

f.close()
sess.close()
