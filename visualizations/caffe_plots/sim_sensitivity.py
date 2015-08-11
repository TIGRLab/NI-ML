from math import sqrt
import numpy as np
from scipy.spatial.distance import dice
from scipy.stats.mstats_basic import mquantiles
from sklearn.utils.extmath import randomized_svd
import tables as tb
import sys
import os
import lmdb
import matplotlib.pyplot as plt
from scipy.stats import describe
from sklearn.manifold import TSNE
from sklearn.decomposition import RandomizedPCA
from plots import visualize_activations, hinton, plot_slices, plot_psa_slices
from sensitivity import sample_binary_perturbation_Sjk, PSA


# Make sure that caffe is on the python path:
caffe_root = '/projects/francisco/repositories/caffe/'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')
import caffe

# GPU mode
caffe.set_device(0)
caffe.set_mode_gpu()

data_path = '/projects/francisco/data/caffe/standardized/HC_sim_cat4_data_2.h5'
model_path = '/projects/francisco/repositories/NI-ML/models/deepnets/caffe/ff/sim/archive/2015-07-30-15-11/'

net_file = 'net.prototxt'
model_file = '_iter_20000.caffemodel'
os.chdir(caffe_root)

input_node = 'train_data_1'
label_node = 'train_classes_1'

data = tb.open_file(data_path, 'r')
test_X = data.get_node('/' + input_node)[:]
test_y = data.get_node('/' + label_node)[:]
data.close()

BATCH_SIZE = 1024
X = test_X[0:BATCH_SIZE]
y = test_y[0:BATCH_SIZE]

os.chdir(model_path)
net = caffe.Net(net_file, model_file, caffe.TEST)

input_shape = (BATCH_SIZE, X.shape[1])

# Show all the layers and their shapes:
for i, v in net.blobs.items():
    print i + ' : ' + str(v.data.shape)

# Grab the input layer:
input_layer = net.blobs[input_node]
labels_layer = net.blobs[label_node]
# The current input layer shape:
input_layer.data.shape
# Reshape the input layer to accept the a larger batch of test data:
input_layer.reshape(*input_shape)
labels_layer.reshape(BATCH_SIZE, 1)

# Reshape the remaining layers to match the input batch size we just changed:
net.reshape()

# Show all the layers and their shapes:
for i, v in net.blobs.items():
    print i + ' : ' + str(v.data.shape)

#Do multiple forward passes and collect stats:
#for i in range(5):
#Assign data to the input:
i = 0
input_layer.data[...] = X[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
#Run a forward pass:
out = net.forward()

# Set up some indices for the classes:

inds = {}

for i in range(4):
    inds[i] = np.where(y == i)[0]

for key, value in inds.items():
    print 'label {}: {} entries'.format(key, len(value))

DX = sample_binary_perturbation_Sjk(net, X)
rx0 = DX[:, :, 0]
rx1 = DX[:, :, 1]
rx2 = DX[:, :, 2]
rx3 = DX[:, :, 3]

sx0 = np.mean(rx0, axis=0) ** 2
sx1 = np.mean(rx1, axis=0) ** 2
sx2 = np.mean(rx2, axis=0) ** 2
sx3 = np.mean(rx3, axis=0) ** 2

# Visualize using the mappings:
mappings = tb.open_file('/projects/francisco/data/caffe/standardized/data_mappings.h5', 'r')
baseline_mask = mappings.get_node('/r_datamask')[:]
volmask = mappings.get_node('/r_volmask')[:]
mappings.close()

baseline_shape = volmask.shape

slices = [(sx0, '0'), (sx1, '1'), (sx2, '2'), (sx3, '3')]
plot_slices(slices, baseline_mask=baseline_mask, baseline_shape=baseline_shape, cols=1)

# PSA:
bcomps0, bevar0, bevar_ratio0 = PSA(rx0, n_components=6)
bcomps1, bevar1, bevar_ratio1 = PSA(rx1, n_components=6)
bcomps2, bevar2, bevar_ratio2 = PSA(rx2, n_components=6)
bcomps3, bevar3, bevar_ratio3 = PSA(rx3, n_components=6)

plot_psa_slices(bcomps0, bevar_ratio0, baseline_mask=baseline_mask, baseline_shape=baseline_shape)
plot_psa_slices(bcomps1, bevar_ratio1, baseline_mask=baseline_mask, baseline_shape=baseline_shape)
plot_psa_slices(bcomps2, bevar_ratio2, baseline_mask=baseline_mask, baseline_shape=baseline_shape)
plot_psa_slices(bcomps3, bevar_ratio3, baseline_mask=baseline_mask, baseline_shape=baseline_shape)