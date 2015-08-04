"""
This script is used to extract features from a pre-trained auto-encoder model by doing a forward pass of an entire
dataset and saving the activations from the auto-encoder's code layer to file.

Usage:
    extract_features.py <netprototxt> <caffemodel> <data_path> <target_file> <datalayer>
"""
import os
import sys
from docopt import docopt
import numpy as np
import tables as tb

caffe_root = '/projects/francisco/repositories/caffe/'
sys.path.insert(0, caffe_root + 'python')

import caffe


def load_data(data_path, input_node):
    data = tb.open_file(data_path, 'r')
    X = data.get_node('/' + input_node)[:]
    data.close()
    return X


def extract_features(net_file, model_file, target_file, data_path, input_node):
    os.chdir(os.path.dirname(net_file))
    net = caffe.Net(net_file, model_file, caffe.TEST)
    
    X = load_data(data_path, input_node)

    BATCH_SIZE = 256
    N = X.shape[0]
    iters = int(np.ceil(N / float(BATCH_SIZE)))

    code_layer = net.blobs['code']
    out_shape = code_layer.data.shape

    X_out = np.zeros(shape=(N, out_shape[1]))

    data_layer = net.blobs.items()[0][1]
    data_layer.reshape(BATCH_SIZE, X.shape[1]) # TODO: only works for 2-D inputs
    net.reshape()

    print 'Extracting features from data...',

    for i in xrange(iters):
        print '.',
        X_b = X[i * BATCH_SIZE: (i+1) * BATCH_SIZE,:]
        data_layer.data[...] = X_b
        net.forward()
        X_out[i * BATCH_SIZE: min((i+1) * BATCH_SIZE, N)] = code_layer.data[0:X_b.shape[0],:].copy()

    np.save(target_file, X_out)
    print 'Saved to {}'.format(target_file)


if __name__ == "__main__":
    arguments = docopt(__doc__)
    target_file = arguments['<target_file>']
    data_path = arguments['<data_path>']
    model_file = arguments['<caffemodel>']
    net_file = arguments['<netprototxt>']
    data_layer = arguments['<datalayer>']

    extract_features(net_file, model_file, target_file, data_path, data_layer)
