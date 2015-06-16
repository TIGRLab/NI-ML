from fuel.schemes import SequentialScheme
from fuel.streams import DataStream
from fuel.transformers import Flatten
import theano
from theano import tensor

from fuel.datasets.hdf5 import H5PYDataset
from blocks.algorithms import GradientDescent, Adam, RMSProp, Scale, AdaDelta
from blocks.algorithms import AdaGrad

from blocks.bricks import MLP, Rectifier, Softmax, Logistic
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.extensions import FinishAfter
from blocks.extensions import Printing
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.extensions.plot import Plot
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_dropout
from blocks.initialization import IsotropicGaussian, Constant
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.roles import WEIGHT, INPUT
from blocks.theano_expressions import l2_norm

# side = b, l, r
side = 'r'
EPOCHS = 50
input_dim = {'l': 11427, 'r': 10519, 'b': 10519 + 11427}
data_file = '/projects/francisco/data/fuel/mci_cn.h5'

if 'b' in side:
    train = H5PYDataset(data_file, which_set='train')
    valid = H5PYDataset(data_file, which_set='valid')
    test = H5PYDataset(data_file, which_set='test')
    x_l = tensor.matrix('l_features')
    x_r = tensor.matrix('r_features')
    x = tensor.concatenate([x_l, x_r], axis=1)

else:
    train = H5PYDataset(data_file, which_set='train', sources=['{}_features'.format(side), 'targets'])
    valid = H5PYDataset(data_file, which_set='valid', sources=['{}_features'.format(side), 'targets'])
    test = H5PYDataset(data_file, which_set='test', sources=['{}_features'.format(side), 'targets'])
    x = tensor.matrix('{}_features'.format(side))

y = tensor.lmatrix('targets')


# Define a feed-forward net with an input, two hidden layers, and a softmax output:
model = MLP(activations=[
    Rectifier(name='input'),
    Rectifier(name='h1'),
    Rectifier(name='h2'),
    Rectifier(name='h3'),
    Softmax(name='output'),
],
            dims=[
                input_dim[side],
                32,
                32,
                16,
                16,
                2],
            weights_init=IsotropicGaussian(),
            biases_init=Constant(0.1))

# Don't forget to initialize params:
model.initialize()

# y_hat is the output of the neural net with x as its inputs
y_hat = model.apply(x)

# Define a cost function to optimize, and a classification error rate.
# Also apply the outputs from the net and corresponding targets:
cost = CategoricalCrossEntropy().apply(y.flatten(), y_hat)
cost.name = 'entropy'
error = MisclassificationRate().apply(y.flatten(), y_hat)
error.name = 'error'

# This is the model: before applying dropout
model = Model(error)

# Need to define the computation graph for the cost func:
cost_graph = ComputationGraph(cost)

# This returns a list of weight vectors for each layer
W = VariableFilter(roles=[WEIGHT])(cost_graph.variables)

# Apply dropout to inputs:
input_graph = ComputationGraph(y_hat)
inputs = VariableFilter([INPUT])(input_graph.variables)
dropout_graph = apply_dropout(input_graph, inputs, 0.2)
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
            batch_size=256)))

training_monitor = TrainingDataMonitoring([cost, error], after_batch=True)

# Use the 'valid' set for validation during training:
validation_stream = Flatten(
    DataStream.default_stream(
        dataset=valid,
        iteration_scheme=SequentialScheme(
            valid.num_examples,
            batch_size=1024)))

validation_monitor = DataStreamMonitoring(
    variables=[cost, error],
    data_stream=validation_stream,
    prefix='validation')

# The main loop will train the network and output reports, etc
main = MainLoop(data_stream=data_stream,
                model=model,
                algorithm=algo,
                extensions=[
                    FinishAfter(after_n_epochs=EPOCHS),
                    Printing(),
                    validation_monitor,
                    training_monitor,
                    Plot('AdniNet_{}'.format(side), channels=[['entropy', 'validation_entropy'],
                                                              ['error', 'validation_error']],
                         after_batch=True)
                ])
main.run()