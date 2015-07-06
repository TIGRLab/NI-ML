import h5py as h5
from fuel.datasets import H5PYDataset
import matplotlib.pyplot as plt
import numpy as np
import theano
from theano import tensor, function
from sklearn.manifold import TSNE
from blocks.serialization import load
from blocks.roles import WEIGHT, INPUT, PARAMETER, OUTPUT
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph


def visualize_activations(cg, layer_output, r_features, l_features, labels):
    act_fn = function(cg.inputs[1:], layer_output)
    activations = act_fn(r_features, l_features)
    tsne = TSNE(n_components=2, random_state=0)
    proj = tsne.fit_transform(activations)
    plt.cla()
    plt.scatter(proj[:, 0], proj[:, 1], c=labels)
    plt.savefig('./{}_layer.png'.format(layer_output.name))
    return proj

data_file = '/projects/francisco/data/fuel/mci_cn.h5'
save_file = '/projects/francisco/repositories/NI-ML/models/deepnets/blocks/pre_trained_lrnet/models/2015-06-29-15:45'
test = H5PYDataset(data_file, which_set='test')
l_x = tensor.matrix('l_features')
r_x = tensor.matrix('r_features')
right_dim = 10519
left_dim = 11427

# Loads the entire loop from last training
main_loop=load(save_file)
model = main_loop.model

# The parameters are in reverse order: output layer params first.
model.parameters

# The individual bricks that make up the model, including any nested bricks (mlp in this case):
bricks = model.get_top_bricks()

# We can find the computational graph for the entire model, then from that
# make a function mapping from inputs to the outputs of the last relu layer:
cg = ComputationGraph([model.outputs[0]])  # Full computational graph
outputs = VariableFilter(roles=[OUTPUT])(cg.variables)  # all layer output variables
final_layer_out = outputs[11]
second_layer_out = outputs[8]
l_input = outputs[1]
r_input = outputs[0]
input_layer_out = tensor.concatenate([l_input, r_input], axis=1)

# These are the graph inputs. Note we don't care about using the targets.
cg.inputs


# Compile a theano function to map from l_features and r_features to the final relu activations,
# doing a forward propagation through the loaded model:
act_fn_final = function(cg.inputs[1:], final_layer_out)

# Fake l, r:
lin = np.random.randint(0, 2, size=(1, left_dim))
rin = np.random.randint(0, 2, size=(1, right_dim))

# Bit pains: theano wants floats.
lin = lin.astype(theano.config.floatX)
rin = rin.astype(theano.config.floatX)

# Activations from random input:
act_fn_final(rin, lin)

# To get activations from all of the input (including candidate samples):
handle = test.open()

# This returns the tuple of l_features, r_features, and labels:
samples = test.get_data(handle, slice(0, test.num_examples))

l_features = samples[0].reshape(-1, left_dim)
r_features = samples[1].reshape(-1, right_dim)
labels = samples[2].reshape(-1, 1)

# Or alternatively use the real fused labels instead:
fused_file = '/projects/francisco/data/fused_test.h5'
fused_data = h5.File(fused_file, 'r')
X=fused_data['test_data_fused'][:]
l_features = X[:, 0:left_dim]
r_features = X[:, left_dim:]
labels = fused_data['test_class_fused'][:]

# We only want MCI-CN right now:
mci_cn_inds = np.where(labels != 0)[0]
l_features = l_features[mci_cn_inds,:]
r_features = r_features[mci_cn_inds,:]
labels = labels[mci_cn_inds]

# labels should be 0, 1 (where 1 will be mci)
labels -= 1
labels.reshape(-1, 1)

# Bit pains: theano wants floats.
l_features = l_features.astype(theano.config.floatX)
r_features = r_features.astype(theano.config.floatX)

# Returns a (samples x hidden unit outputs) matrix of activations:
test_outs_final = act_fn_final(r_features, l_features)

# Visualize all of the hidden layers:
h1 = tensor.concatenate([outputs[2], outputs[3]], axis=1)
h2 = tensor.concatenate([outputs[6], outputs[7]], axis=1)
h1.name = 'h1_apply_output'
h2.name = 'h2_apply_output'
layers_list = [
    h1,
    h2,
    outputs[9],
    outputs[11]
]

projections = []
for layer in layers_list:
    print 'Visualizing {}'.format(layer.name)
    projections.append(visualize_activations(cg, layer, r_features, l_features, labels))

# Visualize the final 2-d layers, pre and post normalization:
for layer in [outputs[-4], outputs[-3]]:
    plt.cla()
    act_fn = function(cg.inputs[1:], layer)
    activations = act_fn(r_features, l_features)
    projections.append(activations)
    plt.scatter(activations[:, 0], activations[:, 1], c=labels)
    plt.savefig('./{}_layer.png'.format(layer.name))