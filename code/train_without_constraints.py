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
model_path="training_without_constraints/"

number_of_positive_examples_x_types=100
number_of_positive_examples_x_partof=100
number_of_negative_examples_x_partof=6000

train_data = get_train_data()
len_of_train_data = len(train_data)
type_of_train_data = get_types_of_train_data()
idx_of_whole_for_train_data = get_partof_of_train_data(train_data)

train_pics = get_pics(train_data)
threshold_side = 6.0/500

existing_types = []
for t in types:
    if t in type_of_train_data:
        existing_types.append(t)

idxs_of_positive_examples_of_types = {}

for t in existing_types:
    idxs_of_positive_examples_of_types[t] = np.array([idx for idx in range(len(type_of_train_data)) if type_of_train_data[idx] == t and
                                                      (train_data[idx][63] - train_data[idx][61] > threshold_side) and
                                                      (train_data[idx][64] - train_data[idx][62] > threshold_side)])


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

# domain definition
objects_of_type = {}
objects_of_type_not = {}
for t in types:
    objects_of_type[t] = ltn.Domain(number_of_positive_examples_x_types,
                                    number_of_features,
                                    label="objects_of_type_"+t)
    objects_of_type_not[t] = ltn.Domain(
        number_of_positive_examples_x_types*(len(types)-2),
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
                       label="examples_of_"+t) for t in existing_types]

clauses_for_negative_examples_of_types = \
    [ltn.Clause([ltn.Literal(False,isOfType[t],[0],[objects_of_type_not[t]])],
                       label="examples_of_not_"+t) for t in types]

clause_for_positive_examples_of_partof = [ltn.Clause([ltn.Literal(True,isPartOf,[0],
                                                                  [object_pairs_in_partOf_relation])],
                                                      label="examples_of_object_pairs_in_partof_relation")]

clause_for_negative_examples_of_partof = [ltn.Clause([ltn.Literal(False, isPartOf, [0],
                                                                  [object_pairs_not_in_partOf_relation])],
                                                     label="examples_of_object_pairs_not_in_part_of_relation")]

KB = ltn.KnowledgeBase("KB_for_pascalpart",
                       clauses_for_positive_examples_of_types +
                       clauses_for_negative_examples_of_types +
                       clause_for_positive_examples_of_partof +
                       clause_for_negative_examples_of_partof,
                       model_path)

def get_feed_dict():
    global idxs_of_positive_examples_of_types, \
        idxs_of_positive_examples_of_partof,\
        idxs_of_negative_examples_of_partof

    feed_dict = {}

    for t in existing_types:
        feed_dict[objects_of_type[t].tensor] = [train_data[idx][1:] for idx in [random.choice(idxs_of_positive_examples_of_types[t]) for _ in range(number_of_positive_examples_x_types)]]

    for t in types:
        feed_dict[objects_of_type_not[t].tensor] = sum([feed_dict[objects_of_type[t1].tensor] for t1 in existing_types if t1 != t],[])

    feed_dict[object_pairs_in_partOf_relation.tensor] = [np.concatenate([train_data[idx[0]][1:],
                                                                         train_data[idx[1]][1:],
                                                                         compute_extra_features(train_data[idx[0]][-4:]*500, train_data[idx[1]][-4:]*500)]) \
                                                         for idx in [random.choice(idxs_of_positive_examples_of_partof) \
                                                                     for _ in range(number_of_positive_examples_x_partof)]]

    feed_dict[object_pairs_not_in_partOf_relation.tensor] = [np.concatenate([train_data[idx[0]][1:],
                                                                         train_data[idx[1]][1:],
                                                                             compute_extra_features(train_data[idx[0]][-4:]*500, train_data[idx[1]][-4:]*500)]) \
                                                             for idx in [random.choice(idxs_of_negative_examples_of_partof) \
                                                                     for _ in range(number_of_negative_examples_x_partof)]]
    return feed_dict

init = tf.initialize_all_variables()
sess = tf.Session(config=config)
sess.run(init) # comment if restore previous model
#KB.restore(sess)
oldval = 0.0
iterations=[]
KB_satiasfiability=[]

f = open(model_path + 'KB_satisfiability_evolution.txt','w')

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
plt.clf()

f.close()
sess.close()
