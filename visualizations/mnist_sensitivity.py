from math import sqrt
import numpy as np
from sklearn.utils.extmath import randomized_svd
import tables as tb
import sys
import os
import lmdb
import matplotlib.pyplot as plt
from scipy.stats import describe
from sklearn.manifold import TSNE
from sklearn.decomposition import RandomizedPCA
from scipy.special import logit
from scipy.special import expit as logistic
from sensitivity import sample_Sjk

# Make sure that caffe is on the python path:
caffe_root = '/home/fran/workspace/caffe/'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')
import caffe

# GPU mode
caffe.set_device(0)
caffe.set_mode_gpu()

# This section for MNIST:
data_path = '/projects/francisco/repositories/caffe/examples/mnist/mnist_test_lmdb/'
model_path = '/projects/francisco/repositories/caffe/examples/mnist'
net_file = 'lenet.prototxt'
model_file = 'lenet_iter_10000.caffemodel'

os.chdir(model_path)


lmdb_env = lmdb.open(data_path)
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()

N = int(lmdb_env.stat()['entries'])
# Convert lmdb data to numpy vectors:
X = np.zeros((N, 28, 28)) # MNIST images are 28 x 28
y = np.zeros((N, 1))
i = 0
for key, value in lmdb_cursor:
    datum = caffe.proto.caffe_pb2.Datum()
    datum.ParseFromString(value)
    label = int(datum.label)
    image = caffe.io.datum_to_array(datum)
    image = image.astype(np.uint8)
    X[i] = image
    y[i] = label
    i += 1

inds = {}

for i in range(10):
    inds[i] = np.where(y==i)[0]

for key, value in inds.items():
    print 'label {}: {} entries'.format(key, len(value))

X_9 = X[inds[9]]
X_0 = X[inds[0]]

# Load the net
net = caffe.Net(net_file, model_file, caffe.TEST)
input_node = 'data'
label_node = 'label'
prediction_node = 'prob'

h = 0.1

#S0 = sample_Sjk(net, X_0, h)