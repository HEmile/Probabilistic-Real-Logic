import logictensornetworks as ltn
import tensorflow as tf
import numpy as np
import csv, pdb
import timeit
import config
import random

data_training_dir = "data/training/"
data_testing_dir = "data/testing/"
zero_distance_threshold = 6
number_of_features = 64

types = np.genfromtxt("data/classes.csv", dtype='U', delimiter=",")

if config.DATASET == 'vehicle':
    selected_types = np.array(['aeroplane','artifact_wing','body','engine','stern','wheel','bicycle','chain_wheel','handlebar','headlight','saddle','bus','bodywork','door','license_plate','mirror','window','car','motorbike','train','coach','locomotive','boat'])
if config.DATASET == 'indoor':
    selected_types = np.array(
        ['bottle', 'body', 'cap', 'pottedplant', 'plant', 'pot', 'tvmonitor', 'screen', 'chair', 'sofa', 'diningtable'])
if config.DATASET == 'animal':
    selected_types = np.array(['person','arm','ear','ebrow','foot','hair','hand','mouth','nose','eye','head','leg','neck','torso','cat','tail','bird','animal_wing','beak','sheep','horn','muzzle','cow','dog','horse','hoof'])
if config.DATASET == 'all':
    selected_types = types[1:]

# Domain containing the objects
objects = ltn.Domain(number_of_features, label="a_bounding_box")

# Domain containing pairs of objects. They get two additional overlap features.
pairs_of_objects = ltn.Domain(2 * number_of_features + 2, label="a_pair_of_bounding_boxes")

# Create type predicates acting on the objects domain
isOfType = {}
for t in selected_types:
    isOfType[t] = ltn.Predicate("is_of_type_" + t, objects, config.TYPE_LAYERS)

# Create partOf predicate acting on the pairs of objects domain
isPartOf = ltn.Predicate("is_part_of", pairs_of_objects, config.PART_OF_LAYERS)

# Create domains for each object type, both inclusive and exclusive
objects_of_type = {}
objects_of_type_not = {}
for t in selected_types:
    objects_of_type[t] = ltn.Domain(number_of_features, label="objects_of_type_" + t)
    objects_of_type_not[t] = ltn.Domain(number_of_features, label="objects_of_type_not_" + t)

object_pairs_in_partOf = ltn.Domain(number_of_features * 2 + 2,
                                    label="object_pairs_in_partof_relation")
object_pairs_not_in_partOf = ltn.Domain(number_of_features * 2 + 2,
                                        label="object_pairs_not_in_partof_relation")


def containment_ratios_between_two_bbxes(bb1, bb2):
    bb1_area = (bb1[-2] - bb1[-4]) * (bb1[-1] - bb1[-3])
    bb2_area = (bb2[-2] - bb2[-4]) * (bb2[-1] - bb2[-3])
    w_intersec = max(0, min([bb1[-2], bb2[-2]]) - max([bb1[-4], bb2[-4]]))
    h_intersec = max(0, min([bb1[-1], bb2[-1]]) - max([bb1[-3], bb2[-3]]))
    bb_area_intersection = w_intersec * h_intersec
    return [float(bb_area_intersection) / bb1_area, float(bb_area_intersection) / bb2_area]


def get_data(train_or_test_swritch, max_rows=10000000):
    assert train_or_test_swritch == "train" or train_or_test_swritch == "test"

    # Fetching the data from the file system

    if train_or_test_swritch == "train":
        data_dir = data_training_dir
    if train_or_test_swritch == "test":
        data_dir = data_testing_dir
    data = np.genfromtxt(data_dir + "features.csv", delimiter=",", max_rows=max_rows)
    types_of_data = types[np.genfromtxt(data_dir + "types.csv", dtype="i", max_rows=max_rows)]

    # Contains the id of the bounding box that is the 'whole' of this object, and -1 if there is no such box.
    idx_whole_for_data = np.genfromtxt(data_dir + "partOf.csv", dtype="i", max_rows=max_rows)

    idx_of_cleaned_data = np.where(np.logical_and(
        np.all(data[:, -2:] - data[:, -4:-2] >= zero_distance_threshold, axis=1),
        np.in1d(types_of_data, selected_types)))[0]
    print("deleting", len(data) - len(idx_of_cleaned_data), "small bb out of", data.shape[0], "bb")
    data = data[idx_of_cleaned_data]
    data[:, -4:] /= 500

    # Cleaning data by removing small bounding boxes and recomputing indexes of partof data

    types_of_data = types_of_data[idx_of_cleaned_data]

    def filter_references(id_references, filtered_ids):
        id_references = id_references[filtered_ids]
        for ii in range(len(id_references)):
            if id_references[ii] != -1 and id_references[ii] in filtered_ids:
                id_references[ii] = np.where(id_references[ii] == filtered_ids)[0]
            else:
                id_references[ii] = -1
        return id_references

    idx_whole_for_data = filter_references(idx_whole_for_data, idx_of_cleaned_data)

    # Grouping bbs that belong to the same picture

    pics = {}
    for i in range(len(data)):
        if data[i][0] in pics:
            pics[data[i][0]].append(i)
        else:
            pics[data[i][0]] = [i]

    if config.RATIO_DATA < 1:
        pics_to_remove = random.sample(pics.keys(), int((1-config.RATIO_DATA)*len(pics.keys())))
        for i in pics_to_remove:
            del pics[i]
        ids_of_selected_data = [i for i in range(len(data)) if data[i][0] in pics]

        data = data[ids_of_selected_data]
        idx_whole_for_data = filter_references(idx_whole_for_data, ids_of_selected_data)
        types_of_data = types_of_data[ids_of_selected_data]

        pics = {}
        for i in range(len(data)):
            if data[i][0] in pics:
                pics[data[i][0]].append(i)
            else:
                pics[data[i][0]] = [i]

    pairs_of_data = np.array(
        [np.concatenate((data[i][1:], data[j][1:], containment_ratios_between_two_bbxes(data[i], data[j]))) for p in
         pics for i in pics[p] for j in pics[p]])

    pairs_of_bb_idxs = np.array([(i, j) for p in pics for i in pics[p] for j in pics[p]])

    partOf_of_pair_of_data = np.array([idx_whole_for_data[i] == j for p in pics for i in pics[p] for j in pics[p]])

    return data, pairs_of_data, types_of_data, partOf_of_pair_of_data, pairs_of_bb_idxs, pics

# Create two dictionaries that contain ontological partOf relations.
def get_part_whole_ontology():
    with open('data/pascalPartOntology.csv') as f:
        ontologyReader = csv.reader(f)
        parts_of_whole = {}
        wholes_of_part = {}
        for row in ontologyReader:
            parts_of_whole[row[0]] = row[1:]
            for t in row[1:]:
                if t in wholes_of_part:
                    wholes_of_part[t].append(row[0])
                else:
                    wholes_of_part[t] = [row[0]]
        for whole in parts_of_whole:
            wholes_of_part[whole] = []
        for part in wholes_of_part:
            if part not in parts_of_whole:
                parts_of_whole[part] = []
    selected_parts_of_whole = {}
    selected_wholes_of_part = {}
    for t in selected_types:
        selected_parts_of_whole[t] = [p for p in parts_of_whole[t] if p in selected_types]
        selected_wholes_of_part[t] = [w for w in wholes_of_part[t] if w in selected_types]
    return selected_parts_of_whole, selected_wholes_of_part


# reporting measures
def precision(conf_matrix, prediction_array=None):
    if prediction_array is not None:
        return conf_matrix.diagonal() / prediction_array
    else:
        return conf_matrix.diagonal() / conf_matrix.sum(1).T


def recall(conf_matrix, gold_array=None):
    if gold_array is not None:
        return conf_matrix.diagonal() / gold_array
    else:
        return conf_matrix.diagonal() / conf_matrix.sum(0)


def f1(precision, recall):
    return np.multiply(2 * precision, recall) / (precision + recall)


print("end of new pascalpart.py")