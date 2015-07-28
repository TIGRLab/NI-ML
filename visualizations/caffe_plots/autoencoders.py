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

# Make sure that caffe is on the python path:
caffe_root = '/projects/francisco/repositories/caffe/'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')
import caffe

# GPU mode
caffe.set_device(0)
caffe.set_mode_gpu()

# This section for MNIST:
data_path = '/projects/francisco/data/caffe/standardized/combined/ad_cn_test.h5'
model_path = '/projects/francisco/repositories/NI-ML/models/deepnets/caffe/shallow_encoder/'
#net_file = 'lenet_train_test.prototxt'
net_file = 'net.prototxt'
model_file = './train/' + '_iter_10000.caffemodel'
os.chdir(caffe_root)

input_node = 'r_hc_features'
label_node = 'label'

data = tb.open_file(data_path, 'r')
test_X = data.get_node('/' + input_node)[:]
test_y = data.get_node('/' + label_node)[:]
data.close()

# Use the first 1024 test samples:
X = test_X[0:1024]
y = test_y[0:1024]

os.chdir(model_path)
net = caffe.Net(net_file, model_file, caffe.TEST)

input_shape = X.shape

# Show all the layers and their shapes:
for i, v in net.blobs.items():
    print i + ' : ' +str(v.data.shape)

# Grab the input layer:
input_layer = net.blobs[input_node]
# The current input layer shape:
input_layer.data.shape
# Reshape the input layer to accept the a larger batch of test data:
input_layer.reshape(*input_shape)

# Reshape the remaining layers to match the input batch size we just changed:
net.reshape()

# Show all the layers and their shapes:
for i, v in net.blobs.items():
    print i + ' : ' +str(v.data.shape)

# Assign data to the input:
input_layer.data[...] = X

# Run a forward pass:
out = net.forward()

# Grab some layer, say, the code layer:
code_layer = net.blobs['code']

# The activations are in the data:
activations = code_layer.data

# ...

# Profit

