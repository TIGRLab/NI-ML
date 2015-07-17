import sys, os
import numpy as np
import tables as tb

net_path = '/projects/francisco/repositories/NI-ML/models/deepnets/caffe/ff/hc/mcicn/right'
caffe_root = '/home/fran/workspace/caffe/'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')
os.chdir(net_path)

data_node = 'r_hc_features'
labels_node = 'label'


import caffe

# GPU mode
caffe.set_device(0)
caffe.set_mode_gpu()

fused_data_file = '/projects/francisco/data/caffe/standardized/standardized_input_data_HC_R_mcicn.h5'
fused_classes_file = '/projects/francisco/data/caffe/standardized/standardized_input_classes_mcicn.h5'

model_file = './train/_iter_30000.caffemodel'
net_file = './net.prototxt'

net = caffe.Net(net_file, model_file, caffe.TEST)

fused_classes = tb.open_file(fused_classes_file, 'r').get_node('/valid_classes_fused')[:]
fused_data = tb.open_file(fused_data_file, 'r').get_node('/valid_data_fused')[:]
BATCH = fused_classes.shape[0]

input_features = net.blobs[data_node].data
net.blobs[data_node].reshape(BATCH, input_features.shape[1])

input_labels = net.blobs[labels_node].data
net.blobs[labels_node].reshape(BATCH, 1)

net.blobs[data_node].data[...] = fused_data
net.blobs[labels_node].data[...]=fused_classes.reshape(BATCH,1)

out = net.forward()
acc=out['accuracy']
loss = out['loss']

print 'Accuracy: {}'.format(acc)
print 'Loss: {}'.format(loss)