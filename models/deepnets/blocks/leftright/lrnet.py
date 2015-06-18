from fuel.datasets import H5PYDataset
from fuel.schemes import SequentialScheme
from fuel.streams import DataStream
from fuel.transformers import Flatten
from theano import tensor
from blocks.algorithms import GradientDescent, RMSProp, Scale
from blocks.bricks import MLP, Rectifier, Softmax, Logistic
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate, BinaryCrossEntropy
from blocks.bricks.parallel import Parallel, Merge
from blocks.extensions import FinishAfter
from blocks.extensions import Printing
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.extensions.plot import Plot
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_dropout
from blocks.initialization import IsotropicGaussian, Constant
from blocks.main_loop import MainLoop
from blocks.roles import WEIGHT, INPUT
from blocks.theano_expressions import l2_norm

data_file = '/projects/francisco/data/fuel/mci_cn.h5'

right_dim = 10519
left_dim = 11427

train = H5PYDataset(data_file, which_set='train')
valid = H5PYDataset(data_file, which_set='valid')
test = H5PYDataset(data_file, which_set='test')

# input features, x, and target classes y, both come from CIFAR dataset.
l_x = tensor.matrix('l_features')
r_x = tensor.matrix('r_features')
y = tensor.lmatrix('targets')


# Inputs -> hidden_1 -> hidden 2
input_mlp = MLP(activations=[
    Rectifier(name='input'),
    Rectifier(name='h1'),
],
                dims=[
                    None,  # Gets set by Parallel brick.
                    16,
                    None,  # Gets set by parallel brick
                ],
                weights_init=IsotropicGaussian(),
                biases_init=Constant(0.01))

# hidden_2 -> hidden_3 -> hidden_4 -> Logistic output
output_mlp = MLP(activations=[
    Rectifier(name='h3'),
    Rectifier(name='h4'),
    Logistic(name='output')],
                 dims=[
                     32,
                     8,
                     8,
                     2,
                 ],
                 weights_init=IsotropicGaussian(),
                 biases_init=IsotropicGaussian())

output_mlp.initialize()

parallel_nets = Parallel(
    input_names=['l_x', 'r_x'],
    input_dims=[left_dim, right_dim],
    output_dims=[16, 16],
    weights_init=IsotropicGaussian(),
    biases_init=IsotropicGaussian(),
    prototype=input_mlp,
)
parallel_nets.initialize()
l_h, r_h = parallel_nets.apply(l_x=l_x, r_x=r_x)

# Concatenate the inputs from the two hidden subnets into a single variable
# for input into the next layer.
merge = tensor.concatenate([l_h, r_h], axis=1)

y_hat = output_mlp.apply(merge)

# Define a cost function to optimize, and a classification error rate:
# Also apply the outputs from the net, and corresponding targets:
cost = CategoricalCrossEntropy().apply(y.flatten(), y_hat)
error = MisclassificationRate().apply(y.flatten(), y_hat)
error.name = 'error'

# Need to define the computation graph:
graph = ComputationGraph(cost)

# This returns a list of weight vectors for each layer
W = VariableFilter(roles=[WEIGHT])(graph.variables)

# Add some regularization to this model:
lam = 0.001
cost += lam * l2_norm(W)
cost.name = 'entropy'

# Apply dropout to inputs:
graph = ComputationGraph(y_hat)
inputs = VariableFilter([INPUT])(graph.variables)
dropout_graph = apply_dropout(graph, inputs, 0.2)
dropout_cost = dropout_graph.outputs[0]

# Learning Algorithm:
algo = GradientDescent(
    step_rule=Scale(learning_rate=0.1),
    params=dropout_graph.parameters,
    cost=cost)

# Data stream used for training model:
data_stream = Flatten(
    DataStream.default_stream(
        dataset=train,
        iteration_scheme=SequentialScheme(
            train.num_examples,
            batch_size=128)))

# Use the 'valid' set for validation during training:
validation_stream = Flatten(
    DataStream.default_stream(
        dataset=valid,
        iteration_scheme=SequentialScheme(
            valid.num_examples,
            batch_size=256)))

monitor = DataStreamMonitoring(
    variables=[cost, error],
    data_stream=validation_stream,
    prefix='validation',
    after_batch=False)

# The main loop will train the network and output reports, etc
main = MainLoop(data_stream=data_stream,
                algorithm=algo,
                extensions=[
                    FinishAfter(after_n_epochs=10),
                    Printing(),
                    monitor,
                    TrainingDataMonitoring([cost, error], after_batch=True),
                    Plot('LR_AdniNet',
                         channels=[
                             ['entropy', 'validation_entropy'],
                             ['error', 'validation_error']],
                         after_batch=True)
                ])
print "Model Initialized, Data Loaded: Start training with main.run()"