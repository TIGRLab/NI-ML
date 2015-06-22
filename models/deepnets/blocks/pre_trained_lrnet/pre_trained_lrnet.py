import ConfigParser
import datetime
from fuel.datasets import H5PYDataset
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream
from fuel.transformers import Flatten
import numpy
import theano
from theano import tensor, function
from blocks.extras.extensions.plot import Plot
import time
from blocks.algorithms import GradientDescent, AdaGrad, RMSProp
from blocks.bricks import Rectifier, MLP, Logistic, Softmax
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.extensions import FinishAfter
from blocks.extensions import Printing
from blocks.extensions.monitoring import TrainingDataMonitoring, DataStreamMonitoring
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.stopping import FinishIfNoImprovementAfter
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_dropout
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.roles import INPUT, OUTPUT, WEIGHT
from blocks.initialization import IsotropicGaussian
from blocks.serialization import load
from blocks.theano_expressions import l2_norm

config = ConfigParser.ConfigParser()
config.readfp(open('./params'))

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
data_file = config.get('hyperparams', 'data_file')

if 'adagrad' in solver:
    solver_type = AdaGrad()
else:
    solver_type = RMSProp(learning_rate=base_lr)

pre_trained_folder = '../ff/models/'

rn_file = '/projects/francisco/repositories/NI-ML/models/deepnets/blocks/ff/models/rnet/2015-06-22-17:45'
ln_file = '/projects/francisco/repositories/NI-ML/models/deepnets/blocks/ff/models/lnet/2015-06-22-17:40'

right_dim = 10519
left_dim = 11427

train = H5PYDataset(data_file, which_set='train')
valid = H5PYDataset(data_file, which_set='valid')
test = H5PYDataset(data_file, which_set='test')

l_x = tensor.matrix('l_features')
r_x = tensor.matrix('r_features')
y = tensor.lmatrix('targets')

lnet = load(ln_file).model.get_top_bricks()[0]
rnet = load(rn_file).model.get_top_bricks()[0]

# Pre-trained layers:

# Inputs -> hidden_1 -> hidden 2
for side, net in zip(['l', 'r'], [lnet, rnet]):
    for child in net.children:
        child.name = side + '_' + child.name

ll1 = lnet.children[0]
lr1 = lnet.children[1]
ll2 = lnet.children[2]
lr2 = lnet.children[3]

rl1 = rnet.children[0]
rr1 = rnet.children[1]
rl2 = rnet.children[2]
rr2 = rnet.children[3]

l_h = lr2.apply(ll2.apply(lr1.apply(ll1.apply(l_x))))
r_h = rr2.apply(rl2.apply(rr1.apply(rl1.apply(r_x))))

# hidden_2 -> hidden_3 -> hidden_4 -> Logistic output
output_mlp = MLP(activations=[
    Rectifier(name='h3'),
    Rectifier(name='h4'),
    Softmax(name='output'),
],
                 dims=[
                     64,
                     8,
                     8,
                     2,
                 ],
                 weights_init=IsotropicGaussian(),
                 biases_init=IsotropicGaussian())

output_mlp.initialize()

# # Concatenate the inputs from the two hidden subnets into a single variable
# # for input into the next layer.
merge = tensor.concatenate([l_h, r_h], axis=1)
#
y_hat = output_mlp.apply(merge)

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

plotting = Plot('AdniNet_LeftRight',
                channels=[
                    ['dropout_entropy', 'validation_entropy'],
                    ['error', 'validation_error'],
                ],
                after_batch=False)

stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M')

# The main loop will train the network and output reports, etc
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
        Checkpoint('./models/{}'.format(stamp))
    ])
main.run()