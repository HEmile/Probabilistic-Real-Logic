#!/usr/bin/env python

from pascalpart import *
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, pdb, math, time
import pprint as pp

path_to_reports = "reports/models_evaluation"

# swith between GPU and CPU
config = tf.ConfigProto(device_count={'GPU': 1})

def plot_confusion_matrix(cm, Xlabels, Ylabels, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(Xlabels))
    plt.xticks(tick_marks, Xlabels, rotation='vertical', size='small')
    plt.yticks(tick_marks, Ylabels, size='small')
    plt.tight_layout()
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')

def unary_predicate_eval(save_path='', decision_threshold=0.5, label='', sess = tf.Session(config=config)):
    number_training_examples = ["%f" % len(idxs_of_types_training_data[t]) for t in types]
    number_training_examples = np.array(number_training_examples, dtype=np.float32) # the background
    predicate_eval = np.array(
        sess.run([ltn.Literal(True, isOfType[t], [0], [objects_of_type_test]).tensor for t in types],
                 feed_dict=feed_dict)).reshape((len(types), len(test_data)))

    # add to background if the argmax is lower to the threshold
    confusion_matrix = np.zeros((len(types) + 1, len(types) + 1))

    for t in types:
        for idx_bb in idxs_of_positive_examples_of_types[t]:

            predicted_label_idx = np.argmax(predicate_eval[:, idx_bb])
            if predicate_eval[predicted_label_idx, idx_bb] >= decision_threshold:
                confusion_matrix[types.index(t), predicted_label_idx] += 1
            else:
                confusion_matrix[types.index(t), len(types)] += 1 # assign to background class

    fig = plt.figure(figsize=(10.0, 10.0))
    cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(cm_normalized, types + ['bkg'], types + ['bkg'], title='CM for types: ')
    #plt.show() #uncomment for show confusion matrix

    fp = confusion_matrix.sum(axis=0)
    fn = confusion_matrix.sum(axis=1)
    prec = np.true_divide(np.diag(confusion_matrix),fp)[0:-1] # remove background class
    rec = np.true_divide(np.diag(confusion_matrix),fn)[0:-1] # remove background class
    f1 = np.true_divide(2*prec*rec, prec + rec)

    mean_prec = np.mean(prec)
    mean_rec = np.mean(rec)
    mean_f1 = np.mean(f1)

    parts, wholes = get_part_whole_ontology()
    part_labels = wholes.keys()
    prec_wholes = []
    rec_wholes = []
    f1_wholes = []
    prec_parts = []
    rec_parts = []
    f1_parts = []

    for t in types:
        if t in part_labels:
            prec_parts.append(prec[types.index(t)])
            rec_parts.append(rec[types.index(t)])
            f1_parts.append(f1[types.index(t)])
        else:
            prec_wholes.append(prec[types.index(t)])
            rec_wholes.append(rec[types.index(t)])
            f1_wholes.append(f1[types.index(t)])

    mean_prec_wholes = np.mean(prec_wholes)
    mean_rec_wholes = np.mean(rec_wholes)
    mean_f1_wholes = np.mean(f1_wholes)
    mean_prec_parts = np.mean(prec_parts)
    mean_rec_parts = np.mean(rec_parts)
    mean_f1_parts = np.mean(f1_parts)

    prec_rec_f1 = np.concatenate(([number_training_examples], [prec], [rec], [f1]), axis=0)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    np.savetxt(os.path.join(save_path, 'CM_types_' + label + '.csv'), confusion_matrix, delimiter=",")
    fig.savefig(os.path.join(save_path, 'CM_types_' + label + '.png'))

    print("th %f --> prec %f, rec %f, f1 %f" % (decision_threshold, mean_prec, mean_rec, mean_f1))
    return [mean_prec, mean_rec, mean_f1]

def binary_predicate_eval(save_path='', decision_threshold=0.5, label='', sess = tf.Session(config=config)):

    predicate_eval = sess.run(ltn.Literal(True, isPartOf, [0], [object_pairs_in_partOf_relation]).tensor,
                                      feed_dict=feed_dict)
    predicate_eval_tensor = tf.convert_to_tensor(predicate_eval[:, 0])

    threshold_tensor = tf.constant(decision_threshold, shape=[len(idxs_of_positive_examples_of_partof)])

    tp_fn = tf.greater_equal(predicate_eval_tensor, threshold_tensor)
    tp_fn_ints = tf.cast(tp_fn, tf.int32)
    tp_tensor = tf.reduce_sum(tp_fn_ints)
    fn_tensor = tf.subtract(tf.shape(predicate_eval_tensor), tp_tensor)[0]
    tp = sess.run(tp_tensor)
    fn = sess.run(fn_tensor)

    predicate_eval = sess.run(ltn.Literal(True, isPartOf, [0], [object_pairs_not_in_partOf_relation]).tensor,
                                      feed_dict=feed_dict)
    predicate_eval_tensor = tf.convert_to_tensor(predicate_eval[:, 0])

    debug = False
    if debug:
        fp = np.where(predicate_eval >= decision_threshold)[0]
        new_types = ['bkg'] + types
        neg_bb = [np.concatenate([test_data[part_whole_pair[0]][1:], test_data[part_whole_pair[1]][1:], compute_extra_features(test_data[part_whole_pair[0]][-4:]*500, test_data[part_whole_pair[1]][-4:]*500)]) for part_whole_pair in idxs_of_negative_examples_of_partof]
        bb_fp = [np.concatenate((neg_bb[int(idx)], [new_types[np.argmax(neg_bb[int(idx)][:60])]],
                                    [new_types[np.argmax(neg_bb[int(idx)][60:120])]])) for idx in fp]
        np.savetxt('bb_fp_types.csv', bb_fp, fmt='%s')

    threshold_tensor = tf.constant(decision_threshold, shape=[len(idxs_of_negative_examples_of_partof)])
    fp_tn = tf.greater_equal(predicate_eval_tensor, threshold_tensor)
    fp_tn_ints = tf.cast(fp_tn, tf.int32)
    fp_tensor = tf.reduce_sum(fp_tn_ints)
    tn_tensor = tf.subtract(tf.shape(predicate_eval_tensor), fp_tensor)[0]
    fp = sess.run(fp_tensor)
    tn = sess.run(tn_tensor)

    prec = np.true_divide(tp, tp + fp)
    rec = np.true_divide(tp, tp + fn)
    f1 = 2 * prec * rec / (prec + rec)

    confusion_matrix = np.array([[tp, fn], [fp, tn]])
    cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    prec_rec_f1 = np.array([[prec], [rec], [f1]])

    fig = plt.figure(figsize=(10.0, 10.0))
    plot_confusion_matrix(cm_normalized, ['pof', 'not pof'], ['pos', 'neg'], title='CM for partOF: ' + label)
    #plt.show() #uncomment to show confusion matrix

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    np.savetxt(os.path.join(save_path, 'CM_partOf_' + label + '.csv'), confusion_matrix, delimiter=",")
    np.savetxt(os.path.join(save_path, 'precRecF1_partOf_' + label + '.csv'), prec_rec_f1, delimiter=",")
    fig.savefig(os.path.join(save_path, 'CM_partOf_' + label + '.png'))
    print("th %f --> prec %f, rec %f, f1 %f, tp %f, fp %f, fn %f, tn %f"%(decision_threshold, prec, rec, f1, tp, fp, fn, tn))
    return [prec, rec, f1]

training_data = get_train_data()
type_of_train_data = get_types_of_train_data()
test_data = get_test_data()
len_of_test_data = len(test_data)
type_of_test_data = get_types_of_test_data()
idx_of_whole_for_test_data = get_partof_of_test_data(test_data)
test_pics = get_pics(test_data)
types.remove("background")

threshold_side = 6.0/500

idxs_of_positive_examples_of_types = {}
idxs_of_negative_examples_of_types = {}
idxs_of_types_training_data = {}

idxs_of_positive_examples_of_partof = [[idx, idx_of_whole_for_test_data[idx]]
                                       for idx in range(len_of_test_data)
                                       if idx_of_whole_for_test_data[idx] >= 0 and
                                       (test_data[idx][63] - test_data[idx][61] > threshold_side) and
                                       (test_data[idx][64] - test_data[idx][62] > threshold_side) and
                                       (test_data[idx_of_whole_for_test_data[idx]][63] - test_data[idx_of_whole_for_test_data[idx]][61] > threshold_side) and
                                       (test_data[idx_of_whole_for_test_data[idx]][64] - test_data[idx_of_whole_for_test_data[idx]][62] > threshold_side)]

idxs_of_negative_examples_of_partof = []

for idx_pic in test_pics:
    for idx_part in test_pics[idx_pic]:
        for idx_whole in test_pics[idx_pic]:
            if idx_of_whole_for_test_data[idx_part] != idx_whole and \
                                    (test_data[idx_part][63] - test_data[idx_part][61] > threshold_side) and \
                                    (test_data[idx_part][64] - test_data[idx_part][62] > threshold_side) and \
                                    (test_data[idx_whole][63] - test_data[idx_whole][61] > threshold_side) and \
                                    (test_data[idx_whole][64] - test_data[idx_whole][62] > threshold_side):
                idxs_of_negative_examples_of_partof.append([idx_part,idx_whole])

for t in types:
    idxs_of_types_training_data[t] = np.array([idx for idx in range(len(training_data)) if type_of_train_data[idx] == t and
                                               (training_data[idx][63] - training_data[idx][61] > threshold_side) and
                                               (training_data[idx][64] - training_data[idx][62] > threshold_side)])

    idxs_of_positive_examples_of_types[t] = np.array([idx for idx in range(len_of_test_data) if type_of_test_data[idx] == t and
                                                      (test_data[idx][63] - test_data[idx][61] > threshold_side) and
                                                      (test_data[idx][64] - test_data[idx][62] > threshold_side)])

    idxs_of_negative_examples_of_types[t] = np.array([idx for idx in range(len_of_test_data) if type_of_test_data[idx] != t and
                                                      (test_data[idx][63] - test_data[idx][61] > threshold_side) and
                                                      (test_data[idx][64] - test_data[idx][62] > threshold_side)])

# Domain definition
objects_of_type = {}
objects_of_type_not = {}

for t in types:
    objects_of_type[t] = ltn.Domain(len(idxs_of_positive_examples_of_types[t]),
                                    number_of_features,
                                    label="objects_of_type_"+t)
    objects_of_type_not[t] = ltn.Domain(len(idxs_of_negative_examples_of_types[t]),
                                        number_of_features,
                                        label="objects_of_type_not_" + t)

object_pairs_in_partOf_relation = ltn.Domain(len(idxs_of_positive_examples_of_partof),
                                                 number_of_features * 2 + 2,
                                                 label="object_pairs_in_partof_relation")
object_pairs_not_in_partOf_relation = ltn.Domain(len(idxs_of_negative_examples_of_partof),
                                                 number_of_features * 2 + 2,
                                                 label="object_pairs_not_in_partof_relation")
objects_of_type_test =  ltn.Domain(len(test_data),
                              number_of_features,
                              label="all_objects_test_set")

# feed_dict filling
feed_dict = {}
for t in types:
    feed_dict[objects_of_type[t].tensor] = [test_data[idx][1:] for idx in idxs_of_positive_examples_of_types[t]]

feed_dict[object_pairs_in_partOf_relation.tensor] = [np.concatenate([test_data[part_whole_pair[0]][1:], test_data[part_whole_pair[1]][1:], compute_extra_features(test_data[part_whole_pair[0]][-4:]*500, test_data[part_whole_pair[1]][-4:]*500)]) for part_whole_pair in idxs_of_positive_examples_of_partof]
feed_dict[object_pairs_not_in_partOf_relation.tensor] = [np.concatenate([test_data[part_whole_pair[0]][1:], test_data[part_whole_pair[1]][1:], compute_extra_features(test_data[part_whole_pair[0]][-4:]*500, test_data[part_whole_pair[1]][-4:]*500)]) for part_whole_pair in idxs_of_negative_examples_of_partof]
feed_dict[objects_of_type_test.tensor] = test_data[:,1:]

# start evaluation

models = ['training_with_constraints/KB_for_pascalpart.ckpt',
          'training_without_constraints/KB_for_pascalpart.ckpt',]

threshold_space = [0.7]
#threshold_space = np.arange(0.5, 0.925, 0.025, dtype=np.float32) #uncomment to perform grid search over several values of threshold
csv_file = open('reports/models_evaluation/types_partOf_models_evaluation.csv','w')
results_writer = csv.writer(csv_file)
results_writer.writerow(['model', 'threshold'])
results_writer.writerow(['']+['types','types','types','partOf','partOf','partOf']*len(threshold_space))

csv_file_types = open('reports/models_evaluation/types_partOf_models_evaluation.csv','w')
types_writer = csv.writer(csv_file_types)
csv_file_partof = open('reports/models_evaluation/types_partOf_models_evaluation.csv','w')
partof_writer = csv.writer(csv_file_partof)


header = ['']
for th in threshold_space:
    header.append('prec th%.3f' % th)
    header.append('rec th%.3f' % th)
    header.append('f1 th%.3f' % th)
    header.append('prec th%.3f'%th)
    header.append('rec th%.3f'%th)
    header.append('f1 th%.3f'%th)

results_writer.writerow(header)

sess = tf.Session(config=config)
saver = tf.train.Saver()
for model in models:
    saver.restore(sess, model)

    th_results = []
    model_label = os.path.split(model)[1].split('.')[0]
    save_path = os.path.join(path_to_reports, os.path.split(model)[0])

    for th in threshold_space:
        print("model %s with threshold %.3f"%(model, th))
        print("types evaluation...")
        unary_pred_results = unary_predicate_eval(save_path, th, model_label, sess=sess)
        print("partOf evaluation...")
        binary_pred_results = binary_predicate_eval(save_path, th, model_label, sess=sess)

        th_results = th_results + unary_pred_results + binary_pred_results

    results_writer.writerow([model] + th_results)

csv_file.close()
sess.close()
