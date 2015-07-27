import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import tables as tb
from sklearn.manifold import TSNE
from scipy.special import logit
from scipy.special import expit as logistic

# Make sure that caffe is on the python path:
caffe_root = '/home/fran/workspace/caffe/'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')
import caffe

# %matplotlib inline
plt.style.use('ggplot')
# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)


def save_activations(set_name, layer_name, net):
    acts = net.blobs[layer_name].data
    np.save('{}.{}.activations.npy'.format(set_name, layer_name), acts)


def visualize_activations(layer_name, activations, labels):
    tsne = TSNE(n_components=2, random_state=0)
    proj = tsne.fit_transform(activations)
    plt.cla()
    ind0 = np.where(labels == 0)[0]
    ind1 = np.where(labels == 1)[0]
    plt.scatter(activations[ind0, 0], activations[ind0, 1], c='mediumturquoise', alpha=0.7)
    plt.scatter(activations[ind1, 0], activations[ind1, 1], c='slategray', alpha=0.7)
    plt.title('{} activations'.format(layer_name))
    plt.show()
    return proj


def get_matrices(data_node, label_node, side, data_file, fused=True):
    if fused:
        node_fused = '_fused'
    else:
        node_fused = ''
    td = tb.open_file(data_file, 'r')
    if 'b' in side:
        left = td.get_node('/' + data_node.format('l') + node_fused)[:]
        right = td.get_node('/' + data_node.format('r') + node_fused)[:]
        test = {'l': left, 'r': right}
    else:
        test = td.get_node('/' + data_node.format(side) + node_fused)[:]
    test_classes = td.get_node('/' + label_node + node_fused)[:]

    return test, test_classes


def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2**np.ceil(np.log(np.abs(matrix).max())/np.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x,y),w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w))
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()
    plt.show()

fused = True
side = 'r'
SIM_DATA = False
path = '/projects/francisco/repositories/NI-ML/models/deepnets/caffe/autoencoder/'
#path = '/projects/francisco/repositories/NI-ML/models/deepnets/caffe/combined/shallow_comb/'
# Sim vs real values:
if SIM_DATA:
    data_file = '/projects/francisco/data/caffe/standardized/HC_sim_cat4_data_2.h5'
    net = path + 'simnet.prototxt'
    model = path + 'train/last/simulated.caffemodel'
    data_node = 'train_data_1'
    label_node = 'train_classes_1'
    set_name = 'simulated'
else:
    data_file = '/projects/francisco/data/caffe/standardized/combined/ad_cn_test.h5'
    net = path + 'deepnet.prototxt'
    #model = path + './archive/2015-07-20/_iter_10000.caffemodel'
    model = path + 'train/_iter_10000.caffemodel'
    data_node = '{}_hc_features'
    label_node = 'label'
    set_name = 'b_hc_ad_cn_shallow_pretrained'

os.chdir(path)

# GPU mode
caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Net(net, model, caffe.TEST)

# Open up some test data:
test, test_classes = get_matrices(data_node, label_node, side, data_file, fused)

# Available layers:
layers = [(k, v.data.shape) for k, v in net.blobs.items()]

# Parameters:
parameters = [(k, v[0].data.shape) for k, v in net.params.items()]

BATCH = min(test_classes.shape[0], 512)

input_labels = net.blobs[label_node].data
net.blobs[label_node].reshape(BATCH, 1)

if not fused:
    if 'sim' in data_file:
        id0 = np.where(test_classes == 0)[0][0:BATCH / 4]
        id1 = np.where(test_classes == 1)[0][0:BATCH / 4]
        id2 = np.where(test_classes == 2)[0][0:BATCH / 4]
        id3 = np.where(test_classes == 3)[0][0:BATCH / 4]
        idall = np.append(id0, [id1, id2, id3])
    else:
        id0 = np.where(test_classes == 0)[0][0:BATCH / 2]
        id1 = np.where(test_classes == 1)[0][0:BATCH / 2]
        idall = np.append(id0, [id1])

    trial_data = test[idall, :]
    trial_classes = test_classes[idall]
else:
    trial_data = test
    trial_classes = test_classes

if 'b' in side:
    net.blobs[data_node.format('l')].reshape(BATCH, trial_data['l'].shape[1])
    net.blobs[data_node.format('r')].reshape(BATCH, trial_data['r'].shape[1])
    net.blobs[data_node.format('l')].data[...] = trial_data['l']
    net.blobs[data_node.format('r')].data[...] = trial_data['r']
else:
    net.blobs[data_node.format(side)].reshape(BATCH, trial_data.shape[1])
    net.blobs[data_node.format(side)].data[...] = trial_data
net.blobs[label_node].data[...] = trial_classes.reshape(BATCH, 1)

#Do a forward pass of batch size size:
out = net.forward()

# Get this batch's labels:
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
for layer in ['encoder1', 'encoder2', 'encoder3']:
    W = np.mat(net.params[layer][0].data)
    b = np.mat(net.params[layer][1].data)
    Z_in = X_in * W.T + b
    visualize_activations(layer, Z_in, labels)
    X_in = np.mat(net.blobs[layer].data)

# Some hinton diags for the last hidden layer weights:
ind5each = np.append(ind0[0:10], [ind1[0:10]])
hinton(W.T[ind5each, 0:20])

#
for layer in ['decoder1', 'decoder2', 'decoder3']:
    visualize_activations(layer, net.blobs[layer].data, labels)
#
# dec1_proj = visualize_activations('encoder1', dec1_acts, labels)
# dec2_proj = visualize_activations('encoder1', dec1_acts, labels)
# #
