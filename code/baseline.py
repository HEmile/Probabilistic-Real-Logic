#!/usr/bin/env python

from pascalpart import *
import pdb
import matplotlib.pyplot as plt
import os

def plot_confusion_matrix(cm, Xlabels, Ylabels, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(Xlabels))
    plt.xticks(tick_marks, Xlabels, rotation='vertical', size='small')
    plt.yticks(tick_marks, Ylabels, size='small')
    plt.tight_layout()
    plt.ylabel('Predicted labels')
    plt.xlabel('True labels')

def unary_predicate_eval(save_path=""):

    types_evaluation = np.zeros(len_of_test_data, dtype=np.int32)
    confusion_matrix = np.zeros((len(types) + 1, len(types) + 1))

    for t in types:
        for idx_bb in idxs_of_positive_examples_of_types[t]:
            predicted_label_idx = np.argmax(test_data[idx_bb,1:-4])
            types_evaluation[idx_bb] = predicted_label_idx -1 # background class is at index -1

            if predicted_label_idx is not 0:
                confusion_matrix[types.index(t), predicted_label_idx -1] += 1
            else:
                confusion_matrix[types.index(t), len(types)] += 1 # assign to background class

    fig = plt.figure(figsize=(10.0, 10.0))
    cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(cm_normalized, types + ['bkg'], types + ['bkg'], title='CM for types eval')
    #plt.show() #uncomment to show confusion matrix

    fp = confusion_matrix.sum(axis=0)
    fn = confusion_matrix.sum(axis=1)
    prec = np.true_divide(np.diag(confusion_matrix),fp)[0:-1] # remove background class
    rec = np.true_divide(np.diag(confusion_matrix),fn)[0:-1] # remove background class
    f1 = np.true_divide(2*prec*rec, prec + rec)

    mean_prec = np.mean(prec)
    mean_rec = np.mean(rec)
    mean_f1 = np.mean(f1)

    # performance for parts and wholes
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

    np.savetxt(os.path.join(save_path, 'CM_types.csv'), confusion_matrix, delimiter=",")
    fig.savefig(os.path.join(save_path, 'CM_types.png'))

    print "prec %f, rec %f, f1 %f" % (mean_prec, mean_rec, mean_f1)
    return [types_evaluation,[mean_prec_parts, mean_rec_parts, mean_f1_parts,
                                 mean_prec_wholes, mean_rec_wholes, mean_f1_wholes,
                                 mean_prec, mean_rec, mean_f1]]

def binary_predicate_eval(testing_bb_pairs, overlap_threshold=0.8, consistency_axioms=False, label=''):

    partof_bb_proposals = []

    # compute overlap of each pair
    for pair in testing_bb_pairs:
        bb_part = test_data[pair[0],-4:]*500 # remove normalization
        bb_whole = test_data[pair[1],-4:]*500

        bb_overlaps = containment_ratios_between_two_bbxes(bb_part, bb_whole)
        part_overlap = bb_overlaps[0]
        whole_overlap = bb_overlaps[1]

        if part_overlap > overlap_threshold and part_overlap > whole_overlap:
            partof_bb_proposals.append((pair[0], pair[1]))

    # compute precision, recall and f1
    tp = set(idxs_of_positive_examples_of_partof).intersection(set(partof_bb_proposals))
    prec = float(len(tp)) / len(partof_bb_proposals)
    rec = float(len(tp)) / len(idxs_of_positive_examples_of_partof)
    f1 = 2 * prec * rec / (prec + rec)

    # types evaluation for every BB and check consistency with axioms
    if consistency_axioms:

        # check consistency with axioms
        part_of_bb = []

        for part_whole in partof_bb_proposals:
            part_label = bb_types_evaluation[part_whole[0]]
            whole_label = bb_types_evaluation[part_whole[1]]

            if part_label != -1 and whole_label != -1:
                if parts.has_key(types[whole_label]):
                    if types[part_label] in parts[types[whole_label]]:
                        part_of_bb.append(tuple(part_whole))

        # compute precision, recall and f1
        tp = set(idxs_of_positive_examples_of_partof).intersection(set(part_of_bb))
        prec = float(len(tp)) / len(part_of_bb)
        rec = float(len(tp)) / len(idxs_of_positive_examples_of_partof)
        f1 = 2 * prec * rec / (prec + rec)

    print "th %f --> prec %f, rec %f, f1 %f"%(overlap_threshold, prec, rec, f1)
    return [prec, rec, f1]

if __name__ == '__main__':

    save_path = "reports/baseline"
    threshold_side = 6.0 / 500

    # import testing bounding boxes and ontology
    test_data = get_test_data()
    len_of_test_data = len(test_data)
    type_of_test_data = get_types_of_test_data()
    idx_of_whole_for_test_data = get_partof_of_test_data(test_data)
    test_pics = get_pics(test_data)
    types.remove("background")
    parts, wholes = get_part_whole_ontology()

    idxs_of_positive_examples_of_types = {}
    for t in types:
        idxs_of_positive_examples_of_types[t] = np.array([idx for idx in range(len_of_test_data) if
                                                          type_of_test_data[idx] == t and
                                                          (test_data[idx][63] - test_data[idx][61] > threshold_side) and
                                                          (test_data[idx][64] - test_data[idx][62] > threshold_side)])

    # compute all pairs of BB per image
    testing_bb_pairs = []

    for idx_pic in test_pics:
        for idx_part in test_pics[idx_pic]:
            for idx_whole in test_pics[idx_pic]:
                if (test_data[idx_part][63] - test_data[idx_part][61] > threshold_side) and\
                    (test_data[idx_part][64] - test_data[idx_part][62] > threshold_side) and \
                    (test_data[idx_whole][63] - test_data[idx_whole][61] > threshold_side) and \
                    (test_data[idx_whole][64] - test_data[idx_whole][62] > threshold_side):
                    testing_bb_pairs.append([idx_part, idx_whole])

    # compute ground truth
    idxs_of_positive_examples_of_partof = [(idx, idx_of_whole_for_test_data[idx])
                                           for idx in range(len_of_test_data)
                                           if idx_of_whole_for_test_data[idx] >= 0 and
                                           (test_data[idx][63] - test_data[idx][61] > threshold_side) and
                                           (test_data[idx][64] - test_data[idx][62] > threshold_side) and
                                           (test_data[idx_of_whole_for_test_data[idx]][63] -
                                            test_data[idx_of_whole_for_test_data[idx]][61] > threshold_side) and
                                           (test_data[idx_of_whole_for_test_data[idx]][64] -
                                            test_data[idx_of_whole_for_test_data[idx]][62] > threshold_side)]

    # types and partOf evaluation
    print "types evaluation..."
    unary_pred_evaluation = unary_predicate_eval(save_path)
    bb_types_evaluation = unary_pred_evaluation[0]
    type_results_to_file = unary_pred_evaluation[1]
    threshold_space = [0.6]

    for th in threshold_space:
		print "partOf with constraints..."
		partOf_constraints_results_to_file = binary_predicate_eval(testing_bb_pairs, th,consistency_axioms=True)

		print "partOf without constraints..."
		partOf_no_constraints_results_to_file = binary_predicate_eval(testing_bb_pairs, th, consistency_axioms=False)

		# write results
		with open(os.path.join(save_path, 'types_partOf_baseline_evaluation.csv'), 'w') as csv_file:
		    results_writer = csv.writer(csv_file)
		    results_writer.writerow(['types'] * 9 + ['partOf wo'] * 3 + ['partOf w'] * 3)
		    results_writer.writerow(['mean prec parts', 'mean rec parts', 'mean f1 parts',
		                             'mean prec wholes', 'mean rec wholes', 'mean f1 wholes',
		                             'mean prec', 'mean rec', 'mean f1',
		                             'prec', 'rec', 'f1','prec', 'rec', 'f1'])

		    results_writer.writerow(type_results_to_file + partOf_no_constraints_results_to_file + partOf_constraints_results_to_file)
