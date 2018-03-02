import logictensornetworks as ltn
import numpy as np
import csv, math, pdb

ltn.default_layers = 6
ltn.default_smooth_factor = 1e-10
ltn.default_tnorm = "goedel"
ltn.default_aggregator = "hmean"
ltn.default_clause_aggregator = "min"

data_training_dir = "data/small_training/"
data_testing_dir = "data/testing/"

with open("data/classes.csv") as f:
    types = f.read().splitlines()
number_of_features = len(types) + 4

def get_train_data(max_rows=0):
    path = data_training_dir+"bbFeaturesTrain.csv"
    if max_rows:
        result = np.genfromtxt(path,delimiter=",",max_rows=max_rows)
    else:
        result = np.genfromtxt(path,delimiter=",")
    for data in result:
        for i in range(1,5):
            data[-i] = data[-i]/500
    return result

def get_test_data(max_rows=0):
    path = data_testing_dir+"bbFeaturesTest.csv"
    if max_rows:
        result = np.genfromtxt(path,delimiter=",",max_rows=max_rows)
    else:
        result = np.genfromtxt(path,delimiter=",")
    for data in result:
        for i in range(1,5):
            data[-i] = data[-i]/500
    return result

def get_types_of_train_data(max_rows=0):
    path = data_training_dir+"bbUnaryPredicatesTrain.txt"
    if max_rows:
        idx_of_type_of_train_data = np.genfromtxt(path,
            delimiter=",",max_rows=max_rows,dtype=np.int)
    else:
        idx_of_type_of_train_data = np.genfromtxt(path,
            delimiter=",",dtype=np.int)
    return np.array([types[idx] for idx in idx_of_type_of_train_data])

def get_types_of_test_data(max_rows=0):
    path = data_testing_dir+"bbUnaryPredicatesTest.txt"
    if max_rows:
        idx_of_type_of_test_data = np.genfromtxt(path,
            delimiter=",",max_rows=max_rows,dtype=np.int)
    else:
        idx_of_type_of_test_data = np.genfromtxt(path,
            delimiter=",",dtype=np.int)
    return np.array([types[idx] for idx in idx_of_type_of_test_data])

def get_partof_of_train_data(train_data):
    return np.genfromtxt(data_training_dir+"bbPartOfTrain.txt",
        dtype=np.int,delimiter=",",max_rows=len(train_data))

def get_partof_of_test_data(test_data):
    return np.genfromtxt(data_testing_dir+"bbPartOfTest.txt",
        dtype=np.int,delimiter=",",max_rows=len(test_data))

def get_pics(data):
    result = {}
    for i in range(len(data)):
        if data[i][0] in result:
            result[data[i][0]].append(i)
        else:
            result[data[i][0]] = [i]
    return result

def get_part_whole_ontology():
    with open('data/pascalPartOntology.csv') as f:
        ontologyReader = csv.reader(f)
        parts = {}
        wholes = {}
        for row in ontologyReader:
            parts[row[0]]=row[1:]
            for t in row[1:]:
                if t in wholes:
                    wholes[t].append(row[0])
                else:
                    wholes[t] = [row[0]]
    return parts, wholes

def compute_extra_features(bb1,bb2):
    bb_area_intersection = 0.0
    bb1_area = (bb1[2] - bb1[0] + 1) * (bb1[3] - bb1[1] + 1)
    bb2_area = (bb2[2] - bb2[0] + 1) * (bb2[3] - bb2[1] + 1)
    w_intersec = np.min([bb1[2], bb2[2]]) - np.max([bb1[0], bb2[0]]) + 1
    h_intersec = np.min([bb1[3], bb2[3]]) - np.max([bb1[1], bb2[1]]) + 1

    if w_intersec > 0 and h_intersec > 0:
        bb_area_intersection = w_intersec * h_intersec

    return [float(bb_area_intersection)/bb1_area, float(bb_area_intersection)/bb2_area]

def containment_ratios_between_two_bbxes(bb1, bb2):
    bb1_area = (bb1[-2] - bb1[-4]) * (bb1[-1] - bb1[-3]) + 1.0/250000
    bb2_area = (bb2[-2] - bb2[-4]) * (bb2[-1] - bb2[-3]) + 1.0/250000
    w_intersec = np.min([bb1[-2], bb2[-2]]) - np.max([bb1[-4], bb2[-4]])
    h_intersec = np.min([bb1[-1], bb2[-1]]) - np.max([bb1[-3], bb2[-3]])

    if w_intersec > 0 and h_intersec > 0:
        bb_area_intersection = w_intersec * h_intersec
    else:
        bb_area_intersection = 0.0

    return [float(bb_area_intersection)/bb1_area, float(bb_area_intersection)/bb2_area]

object = ltn.Domain(1,number_of_features,label="a_generic_object")
pairOfObjects = ltn.Domain(1,2*number_of_features+2,label="a_generic_pair_of_objects")
isOfType = {}

for t in types:
    isOfType[t] = ltn.Predicate("is_a"+t,[object])
isPartOf = ltn.Predicate("is_part_of",[pairOfObjects])
print("end of pascalpart.py")
