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

model_path = '/projects/francisco/repositories/NI-ML/models/deepnets/caffe/autoencoder/archive/2015-08-04-11-47/'
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
net.reshape()
out=net.forward()

X_hat_ae = net.blobs['output_Sigmoid'].data


# Do some PCA vs AE analysis:
# Extract some principal components

pca, X_hat_pca = transform_PCA(code.data.shape[1], train_X, test_fused_X)

# DICE Scores for PCA
D_pca = score_reconstructions(test_fused_X, X_hat_pca)

# DICE Scores for AE
D = score_reconstructions(test_fused_X, X_hat_ae)

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


