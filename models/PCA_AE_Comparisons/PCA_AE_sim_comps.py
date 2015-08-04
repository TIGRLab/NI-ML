import sys
import os
import numpy as np
import tables as tb
from plots import plot_slices
from PCA_Utils import transform_PCA
from PCA_Utils import score_reconstructions

# Make sure that caffe is on the python path:
caffe_root = '/projects/francisco/repositories/caffe/'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')
import caffe

# GPU mode
caffe.set_device(0)
caffe.set_mode_gpu()

input_node = 'train_data_1'
label_node = 'train_classes_1'

data_path = '/projects/francisco/data/caffe/standardized/HC_sim_cat4_data_2.h5'

data = tb.open_file(data_path, 'r')
X = data.get_node('/' + input_node)[:]
y = data.get_node('/' + label_node)[:]
data.close()

model_path = '/projects/francisco/repositories/NI-ML/models/deepnets/caffe/autoencoder/simencoder/archive/2015-07-31-14-23/'

net_file = 'net.prototxt'
model_file = '_iter_30000.caffemodel'
os.chdir(caffe_root)

input_node = 'train_data_1'
label_node = 'train_classes_1'


os.chdir(model_path)
net = caffe.Net(net_file, model_file, caffe.TEST)
code = net.blobs['code']

train_X = X[1024:]
train_y = y[1024:]
test_X = X[0:1024]
test_y = y[0:1024]

test_y = test_y.reshape(1024, 1)


# Run fused test data through AE and grab  reconstructions
input_shape = test_X.shape

# Grab the input layer:
input_layer = net.blobs[input_node]
# The current input layer shape:
input_layer.data.shape
# Reshape the input layer to accept the a larger batch of test data:
input_layer.reshape(*test_X.shape)
input_layer.data[...] = test_X

net.blobs['output_sigmoid'].reshape(*test_X.shape)
net.reshape()

out=net.forward()

X_hat_ae = net.blobs['output_sigmoid'].data


# Do some PCA vs AE analysis:
# Extract some principal components

X_hat_pca = transform_PCA(code.data.shape[1], train_X, test_X)


# DICE Scores for PCA
D_pca = score_reconstructions(test_X, X_hat_pca)

# DICE Scores for AE
D = score_reconstructions(test_X, X_hat_ae)

mappings = tb.open_file('/projects/francisco/data/caffe/standardized/data_mappings.h5', 'r')
baseline_mask = mappings.get_node('/r_datamask')[:]
volmask = mappings.get_node('/r_volmask')[:]
mappings.close()

# Visualize some random slices:
ae_pca = []
for i in range(6):
    i = np.random.random_integers(test_X.shape[0])
    ae_pca.append((X_hat_pca[i], 'PCA X_hat{}'.format(i)))
    ae_pca.append((test_X[i], 'X{}'.format(i)))
    ae_pca.append((X_hat_ae[i], 'AE X_hat {}'.format(i)))

baseline_shape = volmask.shape
plot_list = []
plot_slices(ae_pca, baseline_shape, baseline_mask, binarize=False, cols=3)

