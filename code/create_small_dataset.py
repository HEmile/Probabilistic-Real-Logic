import numpy as np

path_new_images_id = "data/small_training/test.txt"

path_bbFeaturesTrain = "data/training/bbFeaturesTrain.csv"
path_bbPartOfTrain = "data/training/bbPartOfTrain.txt"
path_bbUnaryPredicatesTrain = "data/training/bbUnaryPredicatesTrain.txt"

path_bbFeaturesTrain_small = "data/small_training/bbFeaturesTrain.csv"
path_bbPartOfTrain_small = "data/small_training/bbPartOfTrain.txt"
path_bbUnaryPredicatesTrain_small = "data/small_training/bbUnaryPredicatesTrain.txt"

print('reading data')

new_images_ids = np.genfromtxt(path_new_images_id, delimiter=",")
bbFeaturesTrain = np.genfromtxt(path_bbFeaturesTrain, delimiter=",")
bbPartOfTrain = np.genfromtxt(path_bbPartOfTrain, delimiter=",")
bbUnaryPredicatesTrain = np.genfromtxt(path_bbUnaryPredicatesTrain, delimiter=",")

print('processing data')

mask = np.in1d(bbFeaturesTrain[:, 0], new_images_ids)
new_bb_ids = np.where(mask)[0]

bbFeaturesTrain_small = bbFeaturesTrain[new_bb_ids]
bbUnaryPredicatesTrain_small = bbUnaryPredicatesTrain[new_bb_ids]
bbPartOfTrain_small = bbPartOfTrain[new_bb_ids]

# adjust indexes from old list of bb to the new one
bbPartOfTrain_small_new_ids = [np.where(new_bb_ids == whole)[0][0] if (whole in new_bb_ids) else whole for whole in bbPartOfTrain_small]

print('writing data')

np.savetxt(path_bbFeaturesTrain_small, bbFeaturesTrain_small, delimiter=',')
np.savetxt(path_bbUnaryPredicatesTrain_small, bbUnaryPredicatesTrain_small, fmt='%d')
np.savetxt(path_bbPartOfTrain_small, bbPartOfTrain_small_new_ids, fmt='%d')
