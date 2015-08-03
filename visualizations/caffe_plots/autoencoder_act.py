#from math import sqrt
import h5py as h5
import numpy as np
#from scipy.stats.mstats_basic import mquantiles
#from sklearn.utils.extmath import randomized_svd
import tables as tb
import sys
import os
import argparse
#import lmdb
#import matplotlib.pyplot as plt
#from scipy.stats import describe
#from sklearn.manifold import TSNE
#from sklearn.decomposition import RandomizedPCA
#from activations import visualize_activations, hinton

print 'start'
parser = argparse.ArgumentParser(description='compute test set activations')
parser.add_argument('-r','--model_root_dir', help='root dir for model (autoencoder_scinet)', required=True)
parser.add_argument('-m','--model_trained_file', help='trained_model (autoencoder_scinet/train/_iter*)', required=True)
parser.add_argument('-c','--model_config_file', help='autoencoder_scinet/net.prototxt file', required=True)
parser.add_argument('-d','--dataset', help='input dataset name', required=True) 
parser.add_argument('-n','--samples', help='no_of_samples', required=True) 
parser.add_argument('-o','--out_file', help='out_file', required=True) 
args = vars(parser.parse_args())

model_dir=args['model_root_dir']
model_trained_file=args['model_trained_file']
model_config_file=args['model_config_file']
dataset=args['dataset']
sampx=int(args['samples'])
out_file=args['out_file']

# Make sure that caffe is on the python path:
caffe_root = '/home/m/mchakrav/nikhil/scratch/deep_learning/caffe/'  # this file is expected to be in {caffe_root}/examples
print caffe_root
sys.path.insert(0, caffe_root + 'python')
import caffe

# GPU mode
caffe.set_device(0)
caffe.set_mode_gpu()

data_path = '/bg01/homescinet/m/mchakrav/nikhil/scratch/deep_learning/caffe/adni_data/combined/'
model_path = '/home/m/mchakrav/nikhil/scratch/deep_learning/NI-ML/models/deepnets/caffe/'

data_file = data_path + dataset
print data_file
net_file = model_path + model_dir + '/' + model_config_file
print net_file
model_file = model_path + model_dir + '/train/' + model_trained_file
print model_file

os.chdir(caffe_root)

input_node = 'r_hc_features'
label_node = 'label'

data = tb.open_file(data_file, 'r')
test_X = data.get_node('/' + input_node)[:]
test_y = data.get_node('/' + label_node)[:]
data.close()

# Use the first 1024 test samples:
#print np.shape(test_X)
#print np.shape(test_y)
#print sampx
X = test_X[:sampx,:]
y = test_y[:sampx]

os.chdir(model_path+model_dir+'/')
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

#layers = ['encoder1','encoder2','code']
layers_acts=[]
out_data=h5.File(out_file,'a') 
for i, v in net.blobs.items():
    #layers_acts.append(net.blobs[layer_name].data)
    #print net.blobs[layer_name].data.shape
    out_data.create_dataset(i, data=np.asarray(v.data)) 

out_data.close()

# Grab some layer, say, the code layer:
#code_layer = net.blobs['code']

# The activations are in the data:
#activations = code_layer.data

# ...
#reconstruction = net.blobs['']
# Profit

#labels = out[label_node]
#print labels
#print len(layers_acts)
#out_data=h5.File(out_file,'a')
#out_data.create_dataset('layers_acts',data=np.asarray(layers_acts))
#out_data.create_dataset('reconstruction',data=reconstruction)
#out_data.create_dataset('labels',data=labels)
#out_data.close()



# Code layer:
#code = net.blobs['code']
#activations = code.data
#ind0 = np.where(labels == 0)[0]
#ind1 = np.where(labels == 1)[0]
#plt.scatter(activations[ind0, 0], activations[ind0, 1], c='mediumturquoise', alpha=0.7)
#plt.scatter(activations[ind1, 0], activations[ind1, 1], c='slategray', alpha=0.7)
#plt.title('code layer activations')
#plt.show()

# np.save('{}.code_activations.npy'.format(set_name), activations)
# np.save('{}.labels.npy'.format(set_name), labels)
#input_name = net.blobs.items()[0][0]

#X_in = np.mat(net.blobs[input_name].data)
# Let's look at activations from decoder layers:
#for layer in ['encoder1', 'encoder2']:
#    W = np.mat(net.params[layer][0].data)
#    b = np.mat(net.params[layer][1].data)
#    Z_in = X_in * W.T + b
#    visualize_activations(layer, Z_in, labels)
#    X_in = np.mat(net.blobs[layer].data)

# Some hinton diags for the last hidden layer weights:
#ind5each = np.append(ind0[0:10], [ind1[0:10]])
#hinton(W.T[ind5each, 0:20])

#
#X_in = np.mat(net.blobs['code'].data)
#for layer in ['decoder1', 'decoder2']:
#    W = np.mat(net.params[layer][0].data)
#    b = np.mat(net.params[layer][1].data)
#    Z_in = X_in * W.T + b
#    visualize_activations(layer, Z_in, labels)
#    X_in = np.mat(net.blobs[layer].data)
#
# Visualize inputs vs their reconstructions:
#mappings = tb.open_file('/projects/francisco/data/caffe/standardized/data_mappings.h5', 'r')
#baseline_mask = mappings.get_node('/r_datamask')[:]
#volmask = mappings.get_node('/r_volmask')[:]
#mappings.close()

#baseline_shape = volmask.shape

#X_hat = net.blobs['output_Sigmoid'].data
#plot_list = []

#for x in range(6):
#    i = np.random.random_integers(1024)
#    plot_list.append((X[i], 'X {}'.format(i)))
#    plot_list.append((X_hat[i], 'X_hat {}'.format(i)))

#plot_slices(plot_list, baseline_shape, baseline_mask)
