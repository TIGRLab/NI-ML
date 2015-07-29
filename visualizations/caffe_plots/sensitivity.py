from math import sqrt
import numpy as np
from sklearn.utils.extmath import randomized_svd
import tables as tb
import sys
import os
import matplotlib.pyplot as plt
from scipy.stats import describe
from scipy.stats.mstats import mquantiles
from sklearn.manifold import TSNE
from sklearn.decomposition import RandomizedPCA
from scipy.special import logit
from scipy.special import expit as logistic

# Make sure that caffe is on the python path:
caffe_root = '/home/fran/workspace/caffe/'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')
import caffe

# GPU mode
caffe.set_device(0)
caffe.set_mode_gpu()


def PSA(X, n_components, iterated_power=3, random_state=None, whiten=False):
    """
    Perform principal sensitivity analysis on X, a matrix of sensitivity maps for each sample vector.

    See: http://arxiv.org/pdf/1412.6785v2.pdf
    :param X:
    :param n_components:
    :param iterated_power:
    :param random_state:
    :param whiten:
    :return:
    """
    n_samples = X.shape[0]

    U, S, V = randomized_svd(X, n_components,
                             n_iter=iterated_power,
                             random_state=random_state)

    explained_variance_ = exp_var = (S ** 2) / n_samples
    full_var = np.var(X, axis=0).sum()
    explained_variance_ratio_ = exp_var / full_var

    if whiten:
        components_ = V / S[:, np.newaxis] * sqrt(n_samples)
    else:
        components_ = V
    return components_, explained_variance_, explained_variance_ratio_


def sample_binary_FF_Sjk(net, X, layerlist):
    """
    Find the sensitivity maps for each unit in the hidden layers by performing
    binary perturbations of each input sample from X.
    :param net:
    :param X:
    :return: The sensitivity maps, plus the collected activations for each sample and layer. (N, D, H_i)
    """

    def matrix_sen(layer):
        ff = net.blobs[layer].data.copy()
        return ff

    DX = {}
    for layer in layerlist:
        shape = net.blobs[layer].data.shape
        DX[layer] = np.zeros([X.shape[0], X.shape[1], shape[1]])
    for i, x in enumerate(X):
        I = np.identity(x.shape[0])
        Forward(net, np.clip(x + I, 0, 1))
        for layer in layerlist:
            DX[layer][i] = matrix_sen(layer)
        Forward(net, np.clip(x - I, 0, 1))
        for layer in layerlist:
            DX[layer][i] = DX[layer][i] - matrix_sen(layer)
    return DX


def sample_binary_perturbation_Sjk(net, X, pred_layer=None):
    """
    Produces sensitivity maps for all prediction classes by perturbing all elements
    of X to find df(x) / dx_i. Perturbations are binary.
    :param net:
    :param X:
    :return: Collected df(x) / dx_i for all samples in X: (N samples, D elements, C classes)
    """
    if not pred_layer:
        pred_layer = net.blobs.items()[-1][0]

    pred_shape = net.blobs[pred_layer].data.shape

    # Collect DXs here: N samples, D dimensions, C classes
    X_shape = list(X.shape)
    X_shape.append(pred_shape[-1])
    out_shape = tuple(X_shape)
    SK = np.zeros(out_shape)

    for i, x in enumerate(X):
        outp, outn, DX = binary_perturbed_Sjk(net, x)
        SK[i] = DX
    return SK


def sample_Sjk(net, X, sensitivity_factor, pred_layer=None):
    """
    Computes the sensitivities for each input for each class over each input sample from X.
    :param net:
    :param X:
    :param sensitivity_factor:
    :param group:
    :return: The df(x)/dx for all samples/elements/classes: (N samples, D elements, C classes)
    """
    if not pred_layer:
        pred_layer = net.blobs.items()[-1][0]
    pred_shape = net.blobs[pred_layer].data.shape

    X_shape = list(X.shape)
    X_shape.append(pred_shape[-1])
    out_shape = tuple(X_shape)

    SK = np.zeros(out_shape)
    for i, x in enumerate(X):
        x_max = np.max(x)
        odim = x.ndim
        oshape = x.shape
        h = sensitivity_factor * x_max
        if odim > 1:
            x = x.ravel()
        H = np.identity(x.shape[0]) * h
        if odim > 1:
            H = H.reshape(H.shape[0], *oshape)
            x = x.reshape(oshape)
        outp, outn, DX = Sjk(net, x, H, h)
        out_shape = list(oshape)
        out_shape.append(DX.shape[-1])
        SK[i] = DX.reshape(out_shape)
    return SK


def Sjk(net, X, H, h):
    """
    Run sensitivity testing for the given input vector x.
    :param net:
    :param X: A matrix of (N samples, (D feature dims)) inputs.
    :param H: A matrix of (N, (D)) perturbations.
    :return: Outputs from both perturbations, plus resulting df(x)/dx matrix.
    """
    outp = Forward(net, X + H)
    pyp = F(net)

    outn = Forward(net, X - H)
    pyn = F(net)

    p_dx = (pyp - pyn)

    # The derivatives:
    # df(x) / dx = f(x + h ) - f(x - h) / 2 * h
    # Only works for hypercubes (ie. all dimensions equal length):
    # di = np.diag_indices_from(H)
    # h = np.tile(H[di], p_dx.shape[1]).reshape(-1, p_dx.shape[1])
    return outp, outn, p_dx / (2 * h)


def binary_perturbed_Sjk(net, X):
    """
    Finds the changes in outputs of the classifier based on the perturbation of
    a single element in the input.
    :param net:
    :param X:
    :return:
    """
    i = np.identity(X.shape[0])
    # Forward pass on positive perturbations
    outp = Forward(net, np.clip(X + i, 0, 1))
    pyp = F(net)

    # Forward pass on negative perturbations
    outn = Forward(net, np.clip(X - i, 0, 1))
    pyn = F(net)

    p_dx = (pyp - pyn)

    return outp, outn, p_dx


def Forward(net, X):
    """
    Run a forward pass of this net for the given input X.
    :param net:
    :param mu:
    :param H:
    :return:
    """
    init_net(net, X)
    input_node = net.blobs.items()[0][0] # TODO: use net.inputs attribute
    net.blobs[input_node].data[...] = X
    out = net.forward()
    return out


def init_net(net, X):
    """
    Initializes the net's data layer dimensions for the given input.
    :param net:
    :param X:
    :return:
    """
    #print 'Initializing Net to X input dimensions.'
    input_node = net.blobs.items()[0][1] # TODO: use net.inputs attribute
    pred_node = net.blobs.items()[-1][1]
    if 'label' in net.blobs.keys():
        net.blobs['label'].reshape(X.shape[0], 1)
    input_node.reshape(*X.shape)
    pred_node.reshape(X.shape[0], pred_node.data.shape[-1])
    net.init = True


def F(net, pred_layer=None):
    """
    Returns the vector of log probabilities logP(y=group|X,W) for the given net: assumes a forward pass using Forward() has
    already been completed.
    :param net:
    :param pred_layer: The name of the net's prediction layer.
    :return:
    """
    if not pred_layer:
        pred_layer = net.blobs.items()[-1][0] # Assumes the prediction layer is last if not given.
    return np.log(net.blobs[pred_layer].data.copy())


def normed_sensitivities(S):
    """
    Normalize matrix containing sensitivity maps for D classes w/ respect to the largest value across *all*
    classes.
    :param S:
    :return:
    """
    largest = np.max(S)
    return S / largest


def normed_sen(S0, S1):
    """
    Normalize two classes' sensitivity vectors relative to each other.
    :param S0:
    :param S1:
    :return:
    """
    largest = np.max([np.abs(S0), np.abs(S1)])
    return S0 / largest, S1 / largest


def sensitivity_info(S):
    """
    Summary stats on the sensitivity vector.
    :param S:
    :return:
    """
    desc = describe(S)
    print 'Mean: ' + str(desc.mean)
    print 'Variance: ' + str(desc.variance)
    print 'Min-Max: ' + str(desc.minmax)
    print 'Most + Sensitivity: ' + str(np.argmax(S))
    print 'Most - Sensitivity: ' + str(np.argmin(S))


def get3DVol(HC_input, HC_shape, input_mask):
    flatvol = np.zeros(np.prod(HC_shape))
    flatvol[input_mask] = HC_input
    vol = flatvol.reshape(-1, HC_shape[2]).T
    return vol


def plot_slices(slice_list, baseline_shape, baseline_mask, llimit=0.01, ulimit=0.99, num_slices=6, xmin=200, xmax=1600):
    """
    Plot dem slices.
    :param slice_list:
    :param llimit:
    :param ulimit:
    :param num_slices:
    :param xmin:
    :param xmax:
    :return:
    """
    plt.style.use('ggplot')
    plt.figure()
    cols = 2
    rows = num_slices / cols
    plt.cla()
    for j, input in enumerate(slice_list):
        quantiles = mquantiles(input[0], [llimit, ulimit])
        wt_vol = get3DVol(input[0], baseline_shape, baseline_mask)
        plt.subplot(rows, cols, j + 1)
        im = plt.imshow(wt_vol[:, xmin:xmax], cmap=plt.cm.RdBu_r, aspect='auto', interpolation='none', vmin=-.06, vmax=0.06)
        plt.grid()
        plt.title(input[1])
        plt.colorbar()
        im.set_clim(quantiles[0], quantiles[1])
        plt.axis('off')
    plt.show()


def plot_features(FFSen, baseline_shape, baseline_mask, llimit=0.01, ulimit=0.99, num_features=32, xmin=200, xmax=1600):
    """
    Visualize the sensitivity maps for the hidden layer units.
    :param FFSen:
    :param llimit:
    :param ulimit:
    :param num_features:
    :param xmin:
    :param xmax:
    :return:
    """
    cols = 2
    rows = num_features / cols

    plt.style.use('ggplot')
    plt.figure()

    plt.cla()
    for j, input in enumerate(FFSen[0:num_features,:]):
        input = input - np.mean(input, axis=0)
        input = input / np.max(np.abs(input)) + 1e-32
        quantiles = mquantiles(input, [llimit, ulimit])
        wt_vol = get3DVol(input, baseline_shape, baseline_mask)
        plt.subplot(rows, cols, j + 1)
        im = plt.imshow(wt_vol[:, xmin:xmax], cmap=plt.cm.RdBu_r, aspect='auto', interpolation='none', vmin=-0.06, vmax=0.06)
        plt.grid()
        im.set_clim(quantiles[0], quantiles[1])
        plt.axis('off')
    plt.show()


def plot_psa_slices(comps, evars, baseline_shape, baseline_mask, llimit=0.0, ulimit=1.0, xmin=200, xmax=1600):
    """
    Visualize "principal sensitivity maps"
    :param comps:
    :param evars:
    :param llimit:
    :param ulimit:
    :param xmin:
    :param xmax:
    :return:
    """
    plt.style.use('ggplot')
    plt.figure()
    cols = 2
    rows = comps.shape[0] / cols
    plt.cla()
    for j in range(comps.shape[0]):
        quantiles = mquantiles(comps[j], [llimit, ulimit])
        wt_vol = get3DVol(comps[j], baseline_shape, baseline_mask)
        plt.subplot(rows, cols, j + 1)
        im = plt.imshow(wt_vol[:, xmin:xmax], cmap=plt.cm.RdBu_r, aspect='auto', interpolation='none', vmin=-.06, vmax=0.06)
        plt.grid()
        plt.title('Explained Variance: {}'.format(evars[j]))
        plt.colorbar()
        im.set_clim(quantiles[0], quantiles[1])
        plt.axis('off')
    plt.show()
