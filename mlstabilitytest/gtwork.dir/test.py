#!numactl --cpunodebind=0 --membind=0 python
import os
#os.environ['num_intra_threads'] = '12'
#os.environ['num_inter_threads'] = '2'

os.environ['KMP_AFFINITY'] = "granularity=fine,compact,1,0"
os.environ['KMP_BLOCKTIME'] = "0"
os.environ['OMP_NUM_THREADS'] = "2"
#os.environ["KMP_SETTINGS"] = "1"

from sklearn.model_selection import KFold
from os.path import dirname, abspath, join
import json

import mlstabilitytest
from mlstabilitytest.training.ElemNetModel import ElemNet

root_dir = '~/TestStabilityML/mlstabilitytest/'

input_file = "/home/giancarlo_g_trimarchi/TestStabilityML/mlstabilitytest/mp_data/data/hullout.json"

print("Reading input data from {}".format(input_file))

model = ElemNet('Ef')

with open(input_file, 'r') as f:
    input_data = json.load(f)
    print("Preprocessing data")
    features, targets, labels = model.preprocess(input_data)

import tensorflow as tf
from tensorflow.python.client import device_lib


print(f"tf.version.VERSION = {tf.version.VERSION}")
print(f"tf.keras.__version__ = {tf.keras.__version__}")
devices = device_lib.list_local_devices()  # this may allocate all GPU memory ?!
print(f"devices = {[x.name for x in devices]}")

predictions = dict()

kf = KFold(n_splits=5, shuffle=True, random_state=10)

iFold = 0
for train_indices, test_indices in kf.split(features):
    print("Training on fold {}".format(iFold))
    iFold += 1

    features_train = features[train_indices]
    targets_train = targets[train_indices]

    features_test = features[test_indices]
    targets_test = targets[test_indices]
    labels_test = labels[test_indices]

    predictions_this_fold = model.fit_and_predict(Xtrain=features_train,
                                                  Ytrain=targets_train,
                                                  Xtest=features_test)

    predictions = {**predictions, **{labels_test[i]: predictions_this_fold[i] for i, x in enumerate(test_indices)}}


output_file = 'ml_input.json'
print("Training complete, saving predictions to {}".format(output_file))
with open(output_file, 'w') as f:
    json.dump(predictions, f)

