import ConfigParser
import datetime
from fuel.datasets import H5PYDataset
from fuel.schemes import ShuffledScheme, ShuffledExampleScheme
from fuel.streams import DataStream
from fuel.transformers import Flatten
import math
import numpy
import theano
from theano import tensor, function
from blocks.extras.extensions.plot import Plot
import time
from blocks.algorithms import GradientDescent, AdaGrad, RMSProp, CompositeRule, VariableClipping
from blocks.bricks import Rectifier, MLP, Logistic, Softmax
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.extensions import FinishAfter, ProgressBar
from blocks.extensions import Printing
from blocks.extensions.monitoring import TrainingDataMonitoring, DataStreamMonitoring
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.stopping import FinishIfNoImprovementAfter
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_dropout
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.roles import INPUT, OUTPUT, WEIGHT, PARAMETER
from blocks.initialization import IsotropicGaussian
from blocks.serialization import load
from blocks.theano_expressions import l2_norm


def main(job_id, params):

    config = ConfigParser.ConfigParser()
    config.readfp(open('./params'))

    max_epoch = int(config.get('hyperparams', 'max_iter', 100))
    base_lr = float(config.get('hyperparams', 'base_lr', 0.01))
    train_batch = int(config.get('hyperparams', 'train_batch', 256))
    valid_batch = int(config.get('hyperparams', 'valid_batch', 512))
    test_batch = int(config.get('hyperparams', 'valid_batch', 512))
    hidden_units = int(config.get('hyperparams', 'hidden_units', 16))


    W_sd = float(config.get('hyperparams', 'W_sd', 0.01))
    W_mu = float(config.get('hyperparams', 'W_mu', 0.0))
    b_sd = float(config.get('hyperparams', 'b_sd', 0.01))
    b_mu = float(config.get('hyperparams', 'b_mu', 0.0))

    dropout_ratio = float(config.get('hyperparams', 'dropout_ratio', 0.2))
    weight_decay = float(config.get('hyperparams', 'weight_decay', 0.001))
    max_norm = float(config.get('hyperparams', 'max_norm', 100.0))
    solver = config.get('hyperparams', 'solver_type', 'rmsprop')
    data_file = config.get('hyperparams', 'data_file')
    fine_tune = config.getboolean('hyperparams', 'fine_tune')

    # Spearmint optimization parameters:
    if params:
        base_lr = float(params['base_lr'][0])
        dropout_ratio = float(params['dropout_ratio'][0])
        hidden_units = params['hidden_units'][0]
        weight_decay = params['weight_decay'][0]

    if 'adagrad' in solver:
        solver_type = CompositeRule([AdaGrad(learning_rate=base_lr), VariableClipping(threshold=max_norm)])
    else:
        solver_type = CompositeRule([RMSProp(learning_rate=base_lr), VariableClipping(threshold=max_norm)])

    rn_file = '/projects/francisco/repositories/NI-ML/models/deepnets/blocks/ff/models/rnet/2015-06-25-18:13'
    ln_file = '/projects/francisco/repositories/NI-ML/models/deepnets/blocks/ff/models/lnet/2015-06-29-11:45'

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

    input_dim = ll2.output_dim + rl2.output_dim

    # hidden_2 -> hidden_3 -> hidden_4 -> Logistic output
    output_mlp = MLP(activations=[
        Rectifier(name='h3'),
        Rectifier(name='h4'),
        Softmax(name='output'),
    ],
                     dims=[
                         input_dim,
                         hidden_units,
                         hidden_units,
                         2,
                     ],
                     weights_init=IsotropicGaussian(std=W_sd, mean=W_mu),
                     biases_init=IsotropicGaussian(std=W_sd, mean=W_mu))

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
    dropout_graph = apply_dropout(cost_graph, [dropout_inputs[0]], 0.2)
    dropout_graph = apply_dropout(dropout_graph, dropout_inputs[1:], dropout_ratio)
    dropout_cost = dropout_graph.outputs[0]
    dropout_cost.name = 'dropout_entropy'

    # If no fine-tuning of l-r models is wanted, find the params for only
    # the joint layers:
    if fine_tune:
        params_to_update = dropout_graph.parameters
    else:
        params_to_update = VariableFilter([PARAMETER], bricks=output_mlp.children)(cost_graph)

    # Learning Algorithm:
    algo = GradientDescent(
        step_rule=solver_type,
        params=params_to_update,
        cost=dropout_cost)

    # algo.step_rule.learning_rate.name = 'learning_rate'

    # Data stream used for training model:
    training_stream = Flatten(
        DataStream.default_stream(
            dataset=train,
            iteration_scheme=ShuffledScheme(
                train.num_examples,
                batch_size=train_batch)))

    training_monitor = TrainingDataMonitoring([dropout_cost,
                                               aggregation.mean(error),
                                               aggregation.mean(algo.total_gradient_norm)],
                                              after_batch=True)


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
        prefix='validation',
        after_epoch=True)

    test_stream = Flatten(
        DataStream.default_stream(
            dataset=test,
            iteration_scheme=ShuffledScheme(
                test.num_examples,
                batch_size=test_batch)))

    test_monitor = DataStreamMonitoring(
        variables=[error],
        data_stream=test_stream,
        prefix='test',
        after_training=True)

    plotting = Plot('AdniNet_LeftRight',
                    channels=[
                        ['dropout_entropy'],
                        ['error', 'validation_error'],
                    ],
    )

    # Checkpoint class used to save model and log:
    stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M')
    checkpoint = Checkpoint('./models/{}'.format(stamp),
                            save_separately=['model', 'log'],
                            every_n_epochs=1)

    # The main loop will train the network and output reports, etc
    main_loop = MainLoop(
        data_stream=training_stream,
        model=model,
        algorithm=algo,
        extensions=[
            validation_monitor,
            training_monitor,
            plotting,
            FinishAfter(after_n_epochs=max_epoch),
            FinishIfNoImprovementAfter(notification_name='validation_error', epochs=1),
            Printing(),
            ProgressBar(),
            checkpoint,
            test_monitor,
        ])
    main_loop.run()
    ve = float(main_loop.log.last_epoch_row['validation_error'])
    te = float(main_loop.log.last_epoch_row['error'])
    spearmint_loss = ve + math.abs(te - ve)
    print 'Spearmint Loss: {}'.format(spearmint_loss)
    return spearmint_loss

if __name__ == "__main__":
    params = {}
    main(0, params)