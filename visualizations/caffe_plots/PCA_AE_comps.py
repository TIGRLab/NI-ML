import sys
import os

import numpy as np
from scipy.spatial.distance import dice
import tables as tb
from sklearn.decomposition import RandomizedPCA

from plots import plot_slices


def transform_PCA(k, train_X, test_X):
    pca = RandomizedPCA(n_components=k)
    pca.fit(train_X)

    # Transform test data with principal components:
    X_reduced = pca.transform(test_X)

    # Reconstruct:
    X_rec = np.dot(X_reduced, pca.components_)

    # Restore mean:
    X_rec += pca.mean_
    return X_rec


def score_reconstructions(X, X_hat):
    D = []
    for i in range(X.shape[0]):
        score = dice(test_fused_X[i].astype(int), np.round_(X_hat[i], 0).astype(int))
        D.append(score)
    print 'Mean DICE Dissimilarity Score (0.0 is no dissimilarity, 1.0 is total dissimilarity): {} '.format(np.mean(D))
    return D


# Make sure that caffe is on the python path:
caffe_root = '/projects/francisco/repositories/caffe/'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')
import caffe

# GPU mode
caffe.set_device(0)
caffe.set_mode_gpu()

input_node = 'r_hc_features'
label_node = 'label'

data_path = '/projects/francisco/data/caffe/standardized/combined/ad_cn_train.h5'
data_path_test = '/projects/francisco/data/caffe/standardized/combined/ad_cn_test.h5'

data = tb.open_file(data_path, 'r')
train_X = data.get_node('/' + input_node)[:]
train_y = data.get_node('/' + label_node)[:]
data.close()

data = tb.open_file(data_path_test, 'r')
test_X = data.get_node('/' + input_node)[:]
test_y = data.get_node('/' + label_node)[:]
test_fused_X = data.get_node('/' + input_node + '_fused')[:]
test_fused_y = data.get_node('/' + label_node + '_fused')[:]
data.close()

#model_path = '/projects/francisco/repositories/NI-ML/models/deepnets/caffe/shallow_encoder/archive/2015-07-29-11-48/'
model_path = '/scratch/francisco/'
net_file = 'net.prototxt'
model_file = '_iter_20000.caffemodel'
os.chdir(caffe_root)

input_node = 'r_hc_features'
label_node = 'label'


os.chdir(model_path)
net = caffe.Net(net_file, model_file, caffe.TEST)
code = net.blobs['code']

# Run fused test data through AE and grab  reconstructions
input_shape = test_fused_X.shape

# Grab the input layer:
input_layer = net.blobs[input_node]
# The current input layer shape:
input_layer.data.shape
# Reshape the input layer to accept the a larger batch of test data:
input_layer.reshape(*input_shape)

out=net.forward()

X_hat_ae = net.blobs['output_Sigmoid'].data


# Do some PCA vs AE analysis:
# Extract some principal components

X_hat_pca = transform_PCA(code.data.shape[1], train_X, test_fused_X)


# DICE Scores for PCA
D_pca = []
for i in range(test_fused_X.shape[0]):
    score = dice(test_fused_X[i].astype(int), np.round_(X_hat_pca[i], 0).astype(int))
    D_pca.append(score)

# DICE Scores for AE
D = []
for i in range(test_fused_X.shape[0]):
    score = dice(test_fused_X[i].astype(int), np.round_(X_hat_ae[i], 0).astype(int))
    D.append(score)

mappings = tb.open_file('/projects/francisco/data/caffe/standardized/data_mappings.h5', 'r')
baseline_mask = mappings.get_node('/r_datamask')[:]
volmask = mappings.get_node('/r_volmask')[:]
mappings.close()

# Visualize some random slices:
ae_pca = []
for i in range(6):
    i = np.random.random_integers(test_fused_X.shape[0])
    ae_pca.append((X_hat_pca[i], 'PCA X_hat{}'.format(i)))
    ae_pca.append((test_fused_X[i], 'X{}'.format(i)))
    ae_pca.append((X_hat_ae[i], 'AE X_hat {}'.format(i)))

baseline_shape = volmask.shape
plot_list = []
plot_slices(ae_pca, baseline_shape, baseline_mask, binarize=False, cols=3)


# Try PCA with various num_components and find the best DICE score from the resulting reconstructions.
K = []
K_comps = [4, 16, 64, 128, 256, 328, 512, 1024]
for k in K_comps:
    print 'Fitting PCA with {} components'.format(k)
    X_hat_pca = transform_PCA(k, train_X, test_fused_X)
    D = score_reconstructions(test_fused_X, X_hat_pca)
    K.append(np.mean(D))

best_ind = np.argmax(K)
print 'Best DICE score {} from {} components'.format(K[best_ind], K_comps[best_ind])