import numpy as np
import sys, os

caffe_root = '/projects/francisco/repositories/caffe/'
sys.path.insert(0, caffe_root + 'python')

import caffe

# Pre-trained model files
combined_model_root_path = '/projects/francisco/repositories/NI-ML/models/deepnets/caffe/combined/shallow_comb/'
l_model_root_path = '/projects/francisco/repositories/NI-ML/models/deepnets/caffe/ff/hc/adcn/left/archive/2015-07-20/'
r_model_root_path= '/projects/francisco/repositories/NI-ML/models/deepnets/caffe/ff/hc/adcn/right/archive/2015-07-20/'

lnet_file = '_iter_50000.caffemodel'
rnet_file = '_iter_60000.caffemodel'
lnet_proto = 'net.prototxt'
rnet_proto = 'net.prototxt'

os.chdir(l_model_root_path)
lnet = caffe.Net(lnet_proto, lnet_file, caffe.TRAIN)


os.chdir(r_model_root_path)
rnet = caffe.Net(rnet_proto, rnet_file, caffe.TRAIN)


os.chdir(combined_model_root_path)
comb_net = caffe.Net('net.prototxt', caffe.TRAIN)

params = ['ff1', 'ff2']

nets = {
    'l': lnet,
    'r': rnet,
}

for side, net in nets.items():
    for layer in params:
        W = net.params[layer][0].data[...]
        b = net.params[layer][1].data[...]
        comb_net.params['{}_hc_{}'.format(side, layer)][0].data[...] = W
        comb_net.params['{}_hc_{}'.format(side, layer)][1].data[...] = b


comb_net.save('pretrained.caffemodel')

