import sys, os
import numpy as np
import tables as tb

net_path = '/projects/francisco/repositories/NI-ML/models/deepnets/caffe/combined/pretrained/'
#net_path = '/projects/francisco/repositories/NI-ML/models/deepnets/caffe/ff/hc/adcn/left/'
caffe_root = '/home/fran/workspace/caffe/'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')
os.chdir(net_path)

combined = True
#fused = '_fused'
fused = ''
side = 'l'

if combined:
    data_node_l = 'l_hc_features{}'.format(fused)
    data_node_r = 'r_hc_features{}'.format(fused)
else:
    data_node = '{}_hc_features{}'.format(side, fused)

labels_node = 'label{}'.format(fused)


import caffe

# GPU mode
caffe.set_device(0)
caffe.set_mode_gpu()

fused_data_file = '/projects/francisco/data/caffe/standardized/combined/ad_cn_test.h5'
#fused_classes_file = '/projects/francisco/data/caffe/standardized/standardized_input_classes_adcn.h5'

model_file = './train/_iter_40000.caffemodel'
net_file = './net.prototxt'

net = caffe.Net(net_file, model_file, caffe.TEST)

#fused_classes = tb.open_file(fused_classes_file, 'r').get_node('/test_classes_fused')[:]
fused_file = tb.open_file(fused_data_file, 'r')

fused_classes =  fused_file.get_node('/label_fused')[:]
BATCH = fused_classes.shape[0]
if combined:
    fused_data_l = fused_file.get_node('/l_hc_features_fused')[:]
    fused_data_r = fused_file.get_node('/r_hc_features_fused')[:]
    input_features_l = net.blobs[data_node_l].data
    input_features_r = net.blobs[data_node_r].data
    net.blobs[data_node_l].reshape(BATCH, input_features_l.shape[1])
    net.blobs[data_node_r].reshape(BATCH, input_features_r.shape[1])
    net.blobs[data_node_l].data[...] = fused_data_l
    net.blobs[data_node_r].data[...] = fused_data_r
else:
    fused_data = fused_file.get_node('/{}_hc_features_fused'.format(side))[:]
    input_features = net.blobs[data_node].data
    net.blobs[data_node].reshape(BATCH, input_features.shape[1])
    net.blobs[data_node].data[...] = fused_data


input_labels = net.blobs[labels_node].data
net.blobs[labels_node].reshape(BATCH, 1)


net.blobs[labels_node].data[...]=fused_classes.reshape(BATCH,1)

out = net.forward()
acc=out['accuracy']
loss = out['loss']

print 'Accuracy: {}'.format(acc)
print 'Loss: {}'.format(loss)