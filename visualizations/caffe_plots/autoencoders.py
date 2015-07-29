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
from plots import visualize_activations, hinton, plot_slices


# Make sure that caffe is on the python path:
caffe_root = '/projects/francisco/repositories/caffe/'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')
import caffe

# GPU mode
caffe.set_device(0)
caffe.set_mode_gpu()

data_path = '/projects/francisco/data/caffe/standardized/combined/ad_cn_test.h5'
#model_path = '/projects/francisco/repositories/NI-ML/models/deepnets/caffe/shallow_encoder/archive/2015-07-29-11-48/'
model_path = '/scratch/francisco/'
net_file = 'net.prototxt'
model_file = '_iter_20000.caffemodel'
os.chdir(caffe_root)

input_node = 'r_hc_features'
label_node = 'label'

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

# Do multiple forward passes and collect stats:
#for i in range(5):
# Assign data to the input:
i=0
input_layer.data[...] = X[i * BATCH_SIZE: (i+1) * BATCH_SIZE]
# Run a forward pass:
out = net.forward()

# Grab some layer, say, the code layer:
code_layer = net.blobs['code']
# The activations are in the data:
activations = code_layer.data
reconstruction = net.blobs['output_Sigmoid']
labels = y[i * BATCH_SIZE: (i+1) * BATCH_SIZE]

# Code layer:
code = net.blobs['code']

activations = code.data
ind0 = np.where(labels == 0)[0]
ind1 = np.where(labels == 1)[0]
plt.scatter(activations[ind0, 0], activations[ind0, 1], c='mediumturquoise', alpha=0.7)
plt.scatter(activations[ind1, 0], activations[ind1, 1], c='slategray', alpha=0.7)
plt.title('code layer activations')
plt.show()

# np.save('{}.code_activations.npy'.format(set_name), activations)
# np.save('{}.labels.npy'.format(set_name), labels)
input_name = net.blobs.items()[0][0]

# X_in = np.mat(net.blobs[input_name].data)
# # Let's look at activations from decoder layers:
# for layer in ['encoder1', 'encoder2']:
#     W = np.mat(net.params[layer][0].data)
#     b = np.mat(net.params[layer][1].data)
#     Z_in = X_in * W.T + b
#     visualize_activations(layer, Z_in, labels)
#     X_in = np.mat(net.blobs[layer].data)

# Some hinton diags for the last hidden layer weights:
#ind5each = np.append(ind0[0:10], [ind1[0:10]])
#hinton(W.T[ind5each, 0:20])

#
# X_in = np.mat(net.blobs['code'].data)
# for layer in ['encoder1', 'encoder2']:
#     Z = net.blobs[layer].data
#     visualize_activations(layer, Z, labels)
#     # W = np.mat(net.params[layer][0].data)
#     # b = np.mat(net.params[layer][1].data)
#     # Z_in = X_in * W.T + b
#     # visualize_activations(layer, Z_in, labels)
#     # X_in = np.mat(net.blobs[layer].data)
#
# # Visualize inputs vs their reconstructions:
mappings = tb.open_file('/projects/francisco/data/caffe/standardized/data_mappings.h5', 'r')
baseline_mask = mappings.get_node('/r_datamask')[:]
volmask = mappings.get_node('/r_volmask')[:]
mappings.close()

baseline_shape = volmask.shape

X_hat = net.blobs['output_Sigmoid'].data
plot_list = []

#for x in range(6):
for i in [53, 6, 26, 62, 57, 9]:
    #i = np.random.random_integers(BATCH_SIZE)
    plot_list.append((X[i], 'X {}'.format(i)))
    plot_list.append((X_hat[i], 'X_hat {}'.format(i)))

#plot_slices(plot_list, baseline_shape, baseline_mask, binarize=True)
#plot_slices(plot_list, baseline_shape, baseline_mask, binarize=False)

# DICE Scores
D = []
for i in range(X.shape[0]):
    score = dice(X[i].astype(int), np.round_(X_hat[i], 0).astype(int))
    D.append(score)

print 'Mean DICE Dissimilarity Score (0.0 is no dissimilarity, 1.0 is exact similarity): {} '.format(np.mean(D))