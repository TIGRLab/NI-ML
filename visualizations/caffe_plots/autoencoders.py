from math import sqrt
import numpy as np
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
from activations import visualize_activations, hinton


def get3DVol(HC_input, HC_shape, input_mask):
    flatvol = np.zeros(np.prod(HC_shape))
    flatvol[input_mask] = HC_input
    vol = flatvol.reshape(-1, HC_shape[2]).T
    return vol


def plot_slices(slice_list, baseline_shape, baseline_mask, llimit=0.01, ulimit=0.99, xmin=200, xmax=1600):
    """
    Plot dem slices.
    :param slice_list:
    :param llimit:
    :param ulimit:
    :param num_slices:
    :param xmin:
    :param xmax:
    :return:
    """
    num_slices = len(slice_list)
    plt.style.use('ggplot')
    plt.figure()
    cols = 2
    rows = num_slices / cols
    plt.cla()
    for j, input in enumerate(slice_list):
        quantiles = mquantiles(input[0], [llimit, ulimit])
        wt_vol = get3DVol(input[0], baseline_shape, baseline_mask)
        plt.subplot(rows, cols, j + 1)
        im = plt.imshow(wt_vol[:, xmin:xmax], cmap=plt.cm.Reds, aspect='auto', interpolation='none', vmin=-.06, vmax=0.06)
        plt.grid()
        plt.title(input[1])
        plt.colorbar()
        im.set_clim(quantiles[0], quantiles[1])
        plt.axis('off')
    plt.show()


# Make sure that caffe is on the python path:
caffe_root = '/projects/francisco/repositories/caffe/'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')
import caffe

# GPU mode
caffe.set_device(0)
caffe.set_mode_gpu()

data_path = '/projects/francisco/data/caffe/standardized/combined/ad_cn_test.h5'
model_path = '/projects/francisco/repositories/NI-ML/models/deepnets/caffe/shallow_encoder/'
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
reconstruction = net.blobs['']
# Profit

labels = out[label_node]

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

X_in = np.mat(net.blobs[input_name].data)
# Let's look at activations from decoder layers:
for layer in ['encoder1', 'encoder2']:
    W = np.mat(net.params[layer][0].data)
    b = np.mat(net.params[layer][1].data)
    Z_in = X_in * W.T + b
    visualize_activations(layer, Z_in, labels)
    X_in = np.mat(net.blobs[layer].data)

# Some hinton diags for the last hidden layer weights:
#ind5each = np.append(ind0[0:10], [ind1[0:10]])
#hinton(W.T[ind5each, 0:20])

#
X_in = np.mat(net.blobs['code'].data)
for layer in ['decoder1', 'decoder2']:
    W = np.mat(net.params[layer][0].data)
    b = np.mat(net.params[layer][1].data)
    Z_in = X_in * W.T + b
    visualize_activations(layer, Z_in, labels)
    X_in = np.mat(net.blobs[layer].data)

# Visualize inputs vs their reconstructions:
mappings = tb.open_file('/projects/francisco/data/caffe/standardized/data_mappings.h5', 'r')
baseline_mask = mappings.get_node('/r_datamask')[:]
volmask = mappings.get_node('/r_volmask')[:]
mappings.close()

baseline_shape = volmask.shape

X_hat = net.blobs['output_Sigmoid'].data
plot_list = []

for x in range(6):
    i = np.random.random_integers(1024)
    plot_list.append((X[i], 'X {}'.format(i)))
    plot_list.append((X_hat[i], 'X_hat {}'.format(i)))

plot_slices(plot_list, baseline_shape, baseline_mask)