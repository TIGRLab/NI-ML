import ConfigParser
import datetime
from fuel.schemes import SequentialScheme, ShuffledScheme
from fuel.streams import DataStream
from fuel.transformers import Flatten
import theano
from theano import tensor

from fuel.datasets.hdf5 import H5PYDataset
import time
from blocks.algorithms import GradientDescent, Adam, RMSProp, Scale, AdaDelta
from blocks.algorithms import AdaGrad

from blocks.bricks import MLP, Rectifier, Softmax, Logistic
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.extensions import FinishAfter
from blocks.extensions import Printing
from blocks.extras.extensions.plot import Plot
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.stopping import FinishIfNoImprovementAfter
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_dropout
from blocks.initialization import IsotropicGaussian, Constant
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.roles import WEIGHT, INPUT
from blocks.theano_expressions import l2_norm


config = ConfigParser.ConfigParser()
config.readfp(open('./params'))

side = config.get('hyperparams', 'side', 'b')
max_iter = int(config.get('hyperparams', 'max_iter', 100))
base_lr = float(config.get('hyperparams', 'base_lr', 0.01))
train_batch = int(config.get('hyperparams', 'train_batch', 256))
valid_batch = int(config.get('hyperparams', 'valid_batch', 256))
test_batch = int(config.get('hyperparams', 'valid_batch', 256))

W_sd = float(config.get('hyperparams', 'W_sd', 0.01))
W_mu = float(config.get('hyperparams', 'W_mu', 0.0))
W_b = float(config.get('hyperparams', 'W_b', 0.01))
dropout_ratio = float(config.get('hyperparams', 'dropout_ratio', 0.2))
weight_decay = float(config.get('hyperparams', 'weight_decay', 0.001))
solver = config.get('hyperparams', 'solver_type', 'rmsprop')

if 'adagrad' in solver:
    solver_type = AdaGrad()
else:
    solver_type = RMSProp(learning_rate=base_lr)

input_dim = {'l': 11427, 'r': 10519, 'b': 10519 + 11427}
data_file = config.get('hyperparams', 'data_file')

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
    Rectifier(name='h1'),
    Rectifier(name='h2'),
    Softmax(name='output'),
],
            dims=[
                input_dim[side],
                32,
                32,
                2],
            weights_init=IsotropicGaussian(std=W_sd, mean=W_mu),
            biases_init=Constant(W_b))

# Don't forget to initialize params:
model.initialize()

# y_hat is the output of the neural net with x as its inputs
y_hat = model.apply(x)

# Define a cost function to optimize, and a classification error rate.
# Also apply the outputs from the net and corresponding targets:
cost = CategoricalCrossEntropy().apply(y.flatten(), y_hat)
error = MisclassificationRate().apply(y.flatten(), y_hat)
error.name = 'error'

# This is the model: before applying dropout
model = Model(cost)

# Need to define the computation graph for the cost func:
cost_graph = ComputationGraph([cost])

# This returns a list of weight vectors for each layer
W = VariableFilter(roles=[WEIGHT])(cost_graph.variables)

# Add some regularization to this model:
cost += weight_decay * l2_norm(W)
cost.name = 'entropy'

# computational graph with l2 reg
cost_graph = ComputationGraph([cost])

# Apply dropout to inputs:
inputs = VariableFilter([INPUT])(cost_graph.variables)
dropout_inputs = [input for input in inputs if input.name.startswith('linear_')]
dropout_graph = apply_dropout(cost_graph, dropout_inputs, dropout_ratio)
dropout_cost = dropout_graph.outputs[0]
dropout_cost.name = 'dropout_entropy'

# Learning Algorithm:
algo = GradientDescent(
    step_rule=solver_type,
    params=dropout_graph.parameters,
    cost=dropout_cost)

# algo.step_rule.learning_rate.name = 'learning_rate'

# Data stream used for training model:
training_stream = Flatten(
    DataStream.default_stream(
        dataset=train,
        iteration_scheme=ShuffledScheme(
            train.num_examples,
            batch_size=train_batch)))

training_monitor = TrainingDataMonitoring([dropout_cost, error], after_batch=True)

# Use the 'valid' set for validation during training:
validation_stream = Flatten(
    DataStream.default_stream(
        dataset=valid,
        iteration_scheme=ShuffledScheme(
            valid.num_examples,
            batch_size=valid_batch)))

validation_monitor = DataStreamMonitoring(
    variables=[cost, error],
    data_stream=validation_stream,
    prefix='validation')

test_stream = Flatten(
    DataStream.default_stream(
        dataset=test,
        iteration_scheme=ShuffledScheme(
            test.num_examples,
            batch_size=test.num_examples)))

test_monitor = DataStreamMonitoring(
    variables=[error],
    data_stream=test_stream,
    prefix='test'
)


# param_monitor = DataStreamMonitoring(
# variables=[algo.step_rule.learning_rate],
# data_stream=validation_stream,
#     prefix='params')

plotting = Plot('AdniNet_{}'.format(side),
                channels=[
                    ['dropout_entropy', 'validation_entropy'],
                    ['error', 'validation_error'],
                ],
                after_batch=False)

# The main loop will train the network and output reports, etc

stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M')
main = MainLoop(
    data_stream=training_stream,
    model=model,
    algorithm=algo,
    extensions=[
        FinishAfter(after_n_epochs=max_iter),
        FinishIfNoImprovementAfter(notification_name='validation_error', epochs=3),
        Printing(),
        validation_monitor,
        training_monitor,
        test_monitor,
        plotting,
        Checkpoint('./models/{}net/{}'.format(side, stamp))
    ])
main.run()
