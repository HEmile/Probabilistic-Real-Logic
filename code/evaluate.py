from pascalpart import *
from collections import Counter
import csv
import pdb, os
import matplotlib
import config

matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['xtick.labelsize'] = 20
matplotlib.rcParams['ytick.labelsize'] = 20
matplotlib.rcParams['legend.fontsize'] = 18

np.set_printoptions(precision=2)
np.set_printoptions(threshold=np.inf)

# swith between GPU and CPU
tf_config = tf.ConfigProto(device_count={'GPU': 1})

thresholds = np.arange(.00, 1.1, .05)
models_dir = "models/"
results_dir = "results"

# errors_percentage = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
# constraints_choice = ["KB_wc_nr_", "KB_nc_nr_"]
errors_percentage = np.array(config.NOISE_VALUES)

constraints_choice = []
for alg in config.ALGORITHMS:
    constraints_choice.append("KB_" + alg + "_nr_")
paths_to_models = ["baseline"]
labels_of_models = ["baseline"]

for error in errors_percentage:
    for constraints in constraints_choice:
        paths_to_models.append(models_dir + constraints + str(error) + ".ckpt")
        labels_of_models.append("KB_" + constraints + "_" + str(error))

# loading test data
test_data, pairs_of_test_data, types_of_test_data, partOF_of_pairs_of_test_data, pairs_of_bb_idxs_test, pics = get_data(
    "test", max_rows=50000)

# generating and printing some report on the test data
number_of_test_data_per_type = Counter(types_of_test_data)
print(number_of_test_data_per_type)
type_cardinality_array = np.array([number_of_test_data_per_type[t] for t in selected_types])
idxs_for_selected_types = np.concatenate([np.where(types == st)[0] for st in selected_types])
print(idxs_for_selected_types)


# generating new features for box overlapping
def partof_baseline_test(bb_pair_idx, wholes_of_part, threshold=0.7, with_partof_axioms=False):
    type_compatibility = True
    if with_partof_axioms:
        type_compatibility = False
        part_whole_pair = pairs_of_bb_idxs_test[bb_pair_idx]
        type_part = types_of_test_data[part_whole_pair[0]]
        type_whole = types_of_test_data[part_whole_pair[1]]
        if type_whole in wholes_of_part[type_part]:
            type_compatibility = True

    return (pairs_of_test_data[bb_pair_idx][-2] >= max(threshold,
                                                       pairs_of_test_data[bb_pair_idx][-1])) and type_compatibility

def auc(precision, recall):
    idx_recall = np.argsort(recall)
    return np.trapz(np.array(precision)[idx_recall], x=np.array(recall)[idx_recall])


def plot_recovery_chart(thresholds, performance_w, performance_wo, performance_b, label):
    width = 0.03  # the width of the bars
    fig, ax = plt.subplots(figsize=(10.0, 8.0))
    rects1 = ax.bar(thresholds - width, performance_w, width, color='b')
    rects2 = ax.bar(thresholds, performance_wo, width, color='g')
    plt.ylabel('AUC', fontsize=22)
    plt.xlabel('Errors', fontsize=22)
    plt.title('AUC evolution for ' + label, fontsize=25)
    ax.set_xticks(thresholds)
    ax.legend((rects1[0], rects2[0]), ('AUC LTN_prior', 'AUC LTN_expl'))
    plt.axis([-0.05, 0.47, 0.1, 0.85])
    fig.savefig(os.path.join(results_dir, 'AP_' + label + '.png'))


def plot_prec_rec_curve(precisionW, recallW, precisionWO, recallWO, precisionB, recallB, label):
    fig = plt.figure(figsize=(10.0, 8.0))

    label_baseline_legend = 'FRCNN'
    if 'part-of' in label:
        recallB = [0.0, recallB[0]]
        precisionB = [precisionB[0], precisionB[0]]
        label_baseline_legend = 'RBPOF'

    aucW = auc(precisionW, recallW)
    aucWO = auc(precisionWO, recallWO)
    aucB = auc(precisionB, recallB)

    plt.plot(recallW, precisionW, lw=3, color='blue', label='LTN_prior: AUC={0:0.3f}'.format(aucW))
    plt.plot(recallWO, precisionWO, lw=3, color='green', label='LTN_expl: AUC={0:0.3f}'.format(aucWO))
    plt.plot(recallB, precisionB, lw=3, color='red', label=label_baseline_legend + ': AUC={0:0.3f}'.format(aucB))
    plt.xlabel('Recall', fontsize=22)
    plt.ylabel('Precision', fontsize=22)
    plt.title('Precision-Recall curve ' + label.split('_')[1], fontsize=25)
    plt.legend(loc="lower left")

    fig.savefig(os.path.join(results_dir, 'prec_rec_curve_' + label + '.png'))


def confusion_matrix_for_baseline(thresholds, with_partof_axioms=False):
    print("")
    print("computing confusion matrix for the baseline")
    confusion_matrix_for_types = {}
    confusion_matrix_for_pof = {}
    for th in thresholds:
        print(th, " ",)
        confusion_matrix_for_types[th] = np.matrix([[0.0] * len(selected_types)] * len(selected_types))
        for bb_idx in range(len(test_data)):
            for st_idx in range(len(selected_types)):
                st_feature_of_bb_idx = test_data[bb_idx][1 + idxs_for_selected_types[st_idx]]
                if st_feature_of_bb_idx >= th:
                    confusion_matrix_for_types[th][
                        st_idx, np.where(selected_types == types_of_test_data[bb_idx])[0][0]] += 1

        confusion_matrix_for_pof[th] = np.matrix([[0.0, 0.0], [0.0, 0.0]])

        wholes_of_part = {}
        if with_partof_axioms:
            _, wholes_of_part = get_part_whole_ontology()

        for bb_pair_idx in range(len(pairs_of_test_data)):
            if partof_baseline_test(bb_pair_idx, wholes_of_part, with_partof_axioms=with_partof_axioms):
                if partOF_of_pairs_of_test_data[bb_pair_idx]:
                    confusion_matrix_for_pof[th][0, 0] += 1
                else:
                    confusion_matrix_for_pof[th][0, 1] += 1
            else:
                if partOF_of_pairs_of_test_data[bb_pair_idx]:
                    confusion_matrix_for_pof[th][1, 0] += 1
                else:
                    confusion_matrix_for_pof[th][1, 1] += 1

    return confusion_matrix_for_types, confusion_matrix_for_pof


# determining the values of the atoms isOfType[t](bb) and isPartOf(bb1,bb2) for every type t and for every bounding box bb, bb1 and bb2.
def compute_values_atomic_formulas(path_to_model):
    predicted_types_values_tensor = tf.concat([isOfType[t].tensor() for t in selected_types], 1)
    predicted_partOf_value_tensor = ltn.Literal(True, isPartOf, pairs_of_objects).tensor
    saver = tf.train.Saver()
    sess = tf.Session(config=tf_config)
    saver.restore(sess, path_to_model)
    values_of_types = sess.run(predicted_types_values_tensor, {objects.tensor: test_data[:, 1:]})
    values_of_partOf = sess.run(predicted_partOf_value_tensor, {pairs_of_objects.tensor: pairs_of_test_data})
    sess.close()
    return values_of_types, values_of_partOf


# computing confusion matrixes for the prediction of a model
def confusion_matrixes_of_model(path_to_model, thresholds):
    print("")
    print("computing confusion matrix for", path_to_model)
    global test_data, types_of_test_data, partOF_of_pairs_of_test_data, bb_idxs_pairs
    values_of_types, values_of_partOf = compute_values_atomic_formulas(path_to_model)
    confusion_matrix_for_types = {}
    confusion_matrix_for_pof = {}
    # pdb.set_trace()
    for th in thresholds:
        print(th, " ",)
        confusion_matrix_for_types[th] = np.matrix([[0.0] * len(selected_types)] * len(selected_types))
        for bb_idx in range(len(test_data)):
            for st_idx in range(len(selected_types)):
                if values_of_types[bb_idx][st_idx] >= th:
                    confusion_matrix_for_types[th][
                        st_idx, np.where(selected_types == types_of_test_data[bb_idx])[0][0]] += 1
        confusion_matrix_for_pof[th] = np.matrix([[0.0, 0.0], [0.0, 0.0]])
        for bb_pair_idx in range(len(pairs_of_test_data)):
            if values_of_partOf[bb_pair_idx] >= th:
                if partOF_of_pairs_of_test_data[bb_pair_idx]:
                    confusion_matrix_for_pof[th][0, 0] += 1
                else:
                    confusion_matrix_for_pof[th][0, 1] += 1
            else:
                if partOF_of_pairs_of_test_data[bb_pair_idx]:
                    confusion_matrix_for_pof[th][1, 0] += 1
                else:
                    confusion_matrix_for_pof[th][1, 1] += 1

    return confusion_matrix_for_types, confusion_matrix_for_pof


measure_per_type = {}
measure_per_pof = {}

measures = ["prec", "recall", "f1"]

for measure in measures:
    measure_per_pof[measure] = {}
    measure_per_type[measure] = {}

for path_to_model in paths_to_models:
    if path_to_model == "baseline":
        cm_types, cm_pof = confusion_matrix_for_baseline(thresholds, with_partof_axioms=False)
    else:
        cm_types, cm_pof = confusion_matrixes_of_model(path_to_model, thresholds)
    for measure in measures:
        measure_per_type[measure][path_to_model] = {}
        measure_per_pof[measure][path_to_model] = {}
    for th in thresholds:
        measure_per_type["prec"][path_to_model][th] = precision(cm_types[th])
        measure_per_type["recall"][path_to_model][th] = recall(cm_types[th], gold_array=type_cardinality_array)
        measure_per_type["f1"][path_to_model][th] = f1(measure_per_type["prec"][path_to_model][th],
                                                       measure_per_type["recall"][path_to_model][th])
        measure_per_pof["prec"][path_to_model][th] = precision(cm_pof[th])
        measure_per_pof["recall"][path_to_model][th] = recall(cm_pof[th])
        measure_per_pof["f1"][path_to_model][th] = f1(measure_per_pof["prec"][path_to_model][th],
                                                      measure_per_pof["recall"][path_to_model][th])

print("")
print("writing report in file " + os.path.join(results_dir, "report.csv"))
with open(os.path.join(results_dir, "report.csv"), "w") as report:
    writer = csv.writer(report, delimiter=';')
    writer.writerow(
        ["threshold", ""] + [y for x in [[th] * len(measures) * len(paths_to_models) for th in thresholds] for y in x])
    writer.writerow(
        ["measure", ""] + [y for x in [[meas] * len(paths_to_models) for meas in measures] for y in x] * len(
            thresholds))
    writer.writerow(["models", ""] + labels_of_models * len(measures) * len(thresholds))
    writer.writerow(
        ["part of", ""] + [measure_per_pof[measure][mod][th][0, 0] for th in thresholds for measure in measures for mod
                           in paths_to_models])
    writer.writerow(
        ["average x types", ""] + [measure_per_type[measure][mod][th].mean() for th in thresholds for measure in
                                   measures for mod in paths_to_models])
    for t in selected_types:
        writer.writerow([t, number_of_test_data_per_type[t]] + [
            measure_per_type[measure][mod][th][0, np.where(selected_types == t)[0][0]] for th in thresholds for measure
            in measures for mod in paths_to_models])

ltn_performance_pof_w = []
ltn_performance_pof_wo = []
ltn_performance_pof_b = []
ltn_performance_types_w = []
ltn_performance_types_wo = []
ltn_performance_types_b = []


def adjust_prec(precision):
    prec = precision
    for idx_prec in range(len(precision)):
        if np.isnan(precision[idx_prec]):
            prec[idx_prec] = precision[idx_prec - 1]
    return prec

def stat(measures, model_name, index_type=None):
    if index_type:
        return [measures['prec'][model_name][th][0, index_type] for th in
                thresholds], [measures['recall'][model_name][th][0, index_type] for th in
                thresholds]
    else:
        return [measures['prec'][model_name][th][0, 0] for th in
                      thresholds], [measures['recall'][model_name][th][0, 0] for th in
                      thresholds]

for error in errors_percentage:
    ap_types_w = []
    ap_types_wo = []
    ap_types_b = []
    prec_types_w = []
    prec_types_wo = []
    prec_types_b = []
    rec_types_w = []
    rec_types_wo = []
    rec_types_b = []

    path_wc = models_dir + "KB_prior_nr_" + str(error) + ".ckpt"
    path_nc = models_dir + "KB_wc_nr_" + str(error) + ".ckpt"

    precisionW, recallW = stat(measure_per_pof, path_wc)
    precisionWO, recallWO = stat(measure_per_pof, path_nc)
    precisionB_pof, recallB_pof = stat(measure_per_pof, 'baseline')

    precisionW = adjust_prec(precisionW)
    precisionWO = adjust_prec(precisionWO)
    precisionB_pof = adjust_prec(precisionB_pof)

    plot_prec_rec_curve(precisionW, recallW, precisionWO, recallWO, precisionB_pof, recallB_pof,
                        str(int(error * 100)) + '_part-of')

    ltn_performance_pof_w.append(auc(precisionW, recallW))
    ltn_performance_pof_wo.append(auc(precisionWO, recallWO))
    recallB = [0.0, recallB_pof[0]]
    precisionB = [precisionB_pof[0], precisionB_pof[0]]
    ltn_performance_pof_b.append(auc(precisionB, recallB))

    for t in selected_types:
        index_type = np.where(selected_types == t)[0][0]

        precisionW_types, recallW_types = stat(measure_per_type, path_wc, index_type)
        precisionWO_types, recallWO_types = stat(measure_per_type, path_nc, index_type)
        precisionB_types, recallB_types = stat(measure_per_type, 'baseline', index_type)

        prec_types_w.append(precisionW_types)
        prec_types_wo.append(precisionWO_types)
        prec_types_b.append(precisionB_types)
        rec_types_w.append(recallW_types)
        rec_types_wo.append(recallWO_types)
        rec_types_b.append(recallB_types)

        precisionW_types = adjust_prec(precisionW_types)
        precisionWO_types = adjust_prec(precisionWO_types)
        precisionB_types = adjust_prec(precisionB_types)
        plot_prec_rec_curve(precisionW_types, recallW_types, precisionWO_types, recallWO_types, precisionB_types,
                            recallB_types, str(int(error * 100)) + "_" + t)

        ap_types_w.append(auc(precisionW_types, recallW_types))
        ap_types_wo.append(auc(precisionWO_types, recallWO_types))
        ap_types_b.append(auc(precisionB_types, recallB_types))

    plot_prec_rec_curve(np.mean(prec_types_w, axis=0), np.mean(rec_types_w, axis=0),
                        np.mean(prec_types_wo, axis=0), np.mean(rec_types_wo, axis=0),
                        np.mean(prec_types_b, axis=0), np.mean(rec_types_b, axis=0), str(int(error * 100)) + "_types")

    ltn_performance_types_w.append(np.mean(ap_types_w))
    ltn_performance_types_wo.append(np.mean(ap_types_wo))
    ltn_performance_types_b.append(np.mean(ap_types_b))

plot_recovery_chart(errors_percentage, ltn_performance_pof_w, ltn_performance_pof_wo, ltn_performance_pof_b, 'part-of')
plot_recovery_chart(errors_percentage, ltn_performance_types_w, ltn_performance_types_wo, ltn_performance_types_b,
                    'types')