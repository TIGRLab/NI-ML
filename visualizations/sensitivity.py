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


def sample_binary_Sjk(net, X):
    """
    Produces sensitivity maps for class 0 and class 1 by perturbing all elements
    of the given data X to find df(x) / dx_i.
    :param net:
    :param X:
    :return: Mean sensitivity maps and collected df(x) / dx_i for all samples in X.
    """
    Sk0 = np.zeros([X.shape[0], X.shape[1]])
    Sk1 = np.zeros([X.shape[0], X.shape[1]])

    for i, x in enumerate(X):
        outp, outn, ski0, ski1 = binary_Sjk(net, x)
        Sk0[i] = ski0
        Sk1[i] = ski1
    return np.mean(Sk0, axis=0), np.mean(Sk1, axis=0), Sk0, Sk1


def sample_binary_FF_Sjk(net, X):
    """
    Find the sensitivity maps for each unit in the hidden layers by performing
    binary perturbations of each input sample from X.
    :param net:
    :param X:
    :return: The sensitivity maps, plus the collected df(x)/dx for each sample and layer.
    """

    def matrix_sen(layer):
        ff = net.blobs[layer].data.copy()
        return ff

    ff1shape = net.blobs['ff1'].data.shape
    ff2shape = net.blobs['ff2'].data.shape
    FF1 = np.zeros([X.shape[0], X.shape[1], ff1shape[1]])
    FF2 = np.zeros([X.shape[0], X.shape[1], ff2shape[1]])
    for i, x in enumerate(X):
        I = np.identity(x.shape[0])
        F(net, np.clip(x + I, 0, 1))
        ff1p = matrix_sen('ff1')
        ff2p = matrix_sen('ff2')
        F(net, np.clip(x - I, 0, 1))
        ff1n = matrix_sen('ff1')
        ff2n = matrix_sen('ff2')
        FF1[i] = ff1p - ff1n
        FF2[i] = ff2p - ff2n
    return np.mean(FF1, axis=0).T, np.mean(FF2, axis=0).T, FF1, FF2


def binary_Sjk(net, x):
    """
    Finds the change in the output units of the classifier F based on the perturbation of
    a single element in the input, x.
    :param net:
    :param x:
    :return:
    """
    i = np.identity(x.shape[0])
    # Forward pass on positive perturbations
    outp = F(net, np.clip(x + i, 0, 1))
    p0 = Py(net, 0); p1 = 1 - p0

    # Forward pass on negative perturbations
    outn = F(net, np.clip(x - i, 0, 1))
    n0 = Py(net, 0); n1 = 1 - n0
    return outp, outn, (p0 - n0), (p1 - n1)


def sample_Sjk(net, X, sensitivity_factor):
    """
    Computes the sensitivities for each input for each class over each input sample from X.
    :param net:
    :param X:
    :param sensitivity_factor:
    :param group:
    :return:
    """
    pred_node = net.blobs.items()[-1][0]
    pred_shape = net.blobs[pred_node].data.shape
    Sk = np.zeros((X.shape[0], X.shape[1], pred_shape[-1]))
    for i, x in enumerate(X):
        oshape = x.shape
        if x.ndim > 1:
            x = x.ravel()
        H = np.identity(x.shape[0]) * sensitivity_factor
        if x.ndim > 1:
            H = H.reshape(H.shape[0], *oshape)
            x = x.reshape(oshape)
        outp, outn, DX = Sjk(net, x, H)

        Sk[i] = DX
    return Sk


def Sjk(net, X, H):
    """
    Run sensitivity testing for the given input vector x.
    :param net:
    :param X:
    :param H:
    :return: outputs from both perturbations, plus df(x)/dx matrix
    """
    outp = F(net, X + H)
    pyp = Py(net)

    outn = F(net, X - H)
    pyn = Py(net)

    p_dx = (pyp - pyn)

    di = np.diag_indices_from(H)
    h = np.tile(H[di], p_dx.shape[1]).reshape(-1, p_dx.shape[1])
    return outp, outn, p_dx / (2 * h)


def F(net, X):
    """
    Run a forward pass of this net for the given input X.
    :param net:
    :param mu:
    :param H:
    :return:
    """
    input_node = net.blobs.items()[0][0]
    if 'label' in net.blobs.keys():
        net.blobs['label'].reshape(X.shape[0], 1)
    net.blobs[input_node].reshape(*X.shape)
    net.blobs[input_node].data[...] = X
    out = net.forward()
    return out

def Py(net):
    """
    Returns the vector of probabilities P(y=group|X,W) for the given net: assumes F(net,X) has already been called.
    :param net:
    :param group:
    :return:
    """
    prediction_node = net.blobs.items()[-1][0]
    return np.log(net.blobs[prediction_node].data.copy())


def normed_sen(S0, S1):
    """
    Normalize the two classes' sensitivity vectors relative to each other.
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


def plot_features(FFSen, llimit=0.01, ulimit=0.99, num_features=32, xmin=200, xmax=1600):
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
    for j, input in enumerate(FFSen[0:32,:]):
        input = input - np.mean(input, axis=0)
        input = input / np.max(np.abs(input))
        quantiles = mquantiles(input, [llimit, ulimit])
        wt_vol = get3DVol(input, baseline_shape, baseline_mask)
        plt.subplot(rows, cols, j + 1)
        im = plt.imshow(wt_vol[:, xmin:xmax], cmap=plt.cm.RdBu_r, aspect='auto', interpolation='none', vmin=-0.06, vmax=0.06)
        plt.grid()
        im.set_clim(quantiles[0], quantiles[1])
        plt.axis('off')
    plt.show()


def plot_psa_slices(comps, evars, llimit=0.0, ulimit=1.0, xmin=200, xmax=1600):
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
    rows = comps / cols
    plt.cla()
    for j in range(comps):
        quantiles = mquantiles(comps[0], [llimit, ulimit])
        wt_vol = get3DVol(comps[j], baseline_shape, baseline_mask)
        plt.subplot(rows, cols, j + 1)
        im = plt.imshow(wt_vol[:, xmin:xmax], cmap=plt.cm.RdBu_r, aspect='auto', interpolation='none', vmin=-.06, vmax=0.06)
        plt.grid()
        plt.title('Explained Variance: {}'.format(evars[j]))
        plt.colorbar()
        im.set_clim(quantiles[0], quantiles[1])
        plt.axis('off')
    plt.show()
