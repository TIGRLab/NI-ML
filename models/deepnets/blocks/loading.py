from fuel.datasets import H5PYDataset
import numpy as np
import theano
from theano import tensor, function
from blocks.serialization import load
from blocks.roles import WEIGHT, INPUT, PARAMETER, OUTPUT
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph

data_file = '/projects/francisco/data/fuel/mci_cn.h5'
model_file = '/projects/francisco/repositories/NI-ML/models/deepnets/blocks/pre_trained_lrnet/models/example'
test = H5PYDataset(data_file, which_set='test')
l_x = tensor.matrix('l_features')
r_x = tensor.matrix('r_features')
right_dim = 10519
left_dim = 11427

# Loads the entire loop from last training
main_loop=load(model_file)
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

# These are the graph inputs. Note we don't care about using the targets.
cg.inputs

# Compile a theano function to map from l_features and r_features to the final relu activations,
# doing a forward propagation through the loaded model:
act_fn = function(cg.inputs[1:], final_layer_out)

# Fake l, r:
lin = np.random.randint(0, 2, size=(1, left_dim))
rin = np.random.randint(0, 2, size=(1, right_dim))

# Bit pains: theano wants floats.
lin = lin.astype(theano.config.floatX)
rin = rin.astype(theano.config.floatX)

# Activations from random input:
act_fn(rin, lin)

# To get activations from real input, consider:
handle = test.open()
# This returns the tuple of l_features, r_features, and labels:
samples = test.get_data(handle)
l_features = samples[0].reshape(-1, left_dim)
r_features = samples[1].reshape(-1, right_dim)
labels = samples[2].reshape(-1,1)

# Returns a (samples x hidden unit outputs) matrix of activations:
test_outs = act_fn(l_features, r_features)



