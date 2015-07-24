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
    Perform principal sensitivity analysis on the covariance of r(x), which is a vector
    of the classifier function's sensitivity to each input feature.

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



def sample_Sjk(net, X, sensitivity_factor, group):
    """
    Computes the mean sensitivity for each input for each class over each input sample from X.
    :param net:
    :param X:
    :param sensitivity_factor:
    :param group:
    :return:
    """
    Sk = np.zeros([X.shape[0], X.shape[1]])
    for i, x in enumerate(X):
        H = np.identity(x.shape[0]) * x * sensitivity_factor
        outp, outn, ski_0, ski_1 = Sjk(net, x, H, x, x, sensitivity_factor)
        ski = (ski_0, ski_1)
        Sk[i] = ski[group]
    return np.mean(Sk, axis=0), Sk


def Sjk(net, X, H, class_0_mean, class_1_mean, sensitivity_factor):
    """
    Run sensitivity testing for the given input vector x.
    :param net:
    :param X:
    :param H:
    :return:
    """
    outp = F(net, X + H)
    p0 = Py(net, 0); p1 = 1 - p0

    outn = F(net, X - H)
    n0 = Py(net, 0); n1 = 1 - n0

    c0_dx = ((p0 - n0) / 2 * sensitivity_factor * class_0_mean) ** 2
    c1_dx = ((p1 - n1) / 2 * sensitivity_factor * class_1_mean) ** 2
    return outp, outn, c0_dx, c1_dx


def F(net, X):
    """
    Run a forward pass of this net for the given input X.
    :param net:
    :param mu:
    :param H:
    :return:
    """
    layer = 'pred'
    net.blobs[label_node].reshape(X.shape[0], 1)
    net.blobs[input_node].reshape(X.shape[0], X.shape[0])
    net.blobs[input_node].data[...] = X
    out = net.forward()
    return out

def Py(net, group):
    """
    Returns the vector of probabilities P(y=group|X,W) for the given net: assumes F(net,X) has already been called.
    :param net:
    :param group:
    :return:
    """
    return np.log(net.blobs['pred'].data[:, group].copy())


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


def plot_slices(slice_list, llimit=0.01, ulimit=0.99, num_slices=6, xmin=200, xmax=1600):
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
    t_comps = len(bcomps0) + len(bcomps1)
    cols = 2
    rows = t_comps / cols
    plt.cla()
    for j in range(t_comps):
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

# This section for FF nets:
input_node = 'r_hc_features'
label_node = 'label'
model_path = '/projects/francisco/repositories/NI-ML/models/deepnets/caffe/ff/hc/adcn/right/archive/2015-07-20/'
model_file = model_path + '_iter_60000.caffemodel'
net_file = model_path + 'net.prototxt'

data = tb.open_file('/projects/francisco/data/caffe/standardized/combined/ad_cn_train.h5', 'r')
X = data.get_node('/' + input_node)[:]
y = data.get_node('/' + label_node)[:]
data.close()

data = tb.open_file('/projects/francisco/data/caffe/standardized/combined/ad_cn_test.h5', 'r')
X_fused = data.get_node('/' + input_node + '_fused')[:]
X_test = data.get_node('/' + input_node)[:]
y_test = data.get_node('/' + label_node)[:]
y_fused = data.get_node('/' + label_node + '_fused')[:]
data.close()

# This section for MNIST:
# model_path = '/projects/francisco/repositories/caffe/examples/mnist'
# net_file = 'lenet.prototxt'
# model_file = '_iter_10000.caffemodel'

os.chdir(model_path)
net = caffe.Net(net_file, model_file, caffe.TEST)


# Calculate mean input for each class, j:
# xmu_j = sum of all inputs with label j / # of such inputs
ind0 = np.where(y == 0)[0]
ind1 = np.where(y == 1)[0]

mu_0 = np.mean(X[ind0, :], axis=0)
mu_1 = np.mean(X[ind1, :], axis=0)

# Perturbation function, H(j,k) returns a vector of all zeros except the kth element, which
# is 0.1 to 0.01 (h_j) of the kth element of Xmu_j.
sensitivity_factor = 0.1
H0 = np.identity(X.shape[1]) * mu_0 * sensitivity_factor
H1 = np.identity(X.shape[1]) * mu_1 * sensitivity_factor

BATCH = H0.shape[0]

# Sensitivity for kth input for class j is (recall F() is the forward pass of the nn):
# (F(xmu_j + H(j,k)) - F(xmu_j - H(j,k)) / 2h_j
# By limit laws, the above gives dF(xmu_j) / dx_j
outp, outn, s0k, _ = Sjk(net, mu_0, H0, mu_0, mu_1, sensitivity_factor)
outp1, outn1, _, s1k = Sjk(net, mu_1, H1, mu_0, mu_1, sensitivity_factor)

sen0, sen1 = normed_sen(s0k, s1k)

# Mean Class Sensitivity using test data:
find0 = np.where(y_fused == 0)[0]
find1 = np.where(y_fused == 1)[0]

X_fused_0 = X_fused[find0, :]
X_fused_1 = X_fused[find1, :]

sx0, rx0 = sample_Sjk(net, X_fused_0, sensitivity_factor, 0)
sx1, rx1 = sample_Sjk(net, X_fused_1, sensitivity_factor, 1)

mean_sen0, mean_sen1 = normed_sen(sx0, sx1)

# Binary sensitivity over the test set for each class:
# Note: should we only go over the X_fused data for that class?
# If so, then we possibly miss out on learning about counter-examples.
bs0, bs1, brx0, brx1 = sample_binary_Sjk(net, X_fused)

normed_bs0, normed_bs1 = normed_sen(bs0, bs1)


# FF1 and FF2 activations:
SFF1, SFF2, FF1, FF2 = sample_binary_FF_Sjk(net, X_fused)

# Principal Sensitivity Analysis
# K = E[r(x)r(x).T] where r(x) is a k-dimensional vector of the partial derivatives of F(x) w/ respect to x_k
# Note that E[r(x)] is already computed above for each class, taking the expectation over the test data.

# On test sample sensitivities:
#comps0, evar0, evar_ratio0 = PSA(rx0, 3)
#comps1, evar1, evar_ratio1 = PSA(rx1, 3)

# On Binary test sample sensitivities:
bcomps0, bevar0, bevar_ratio0 = PSA(brx0, n_components=6)
bcomps1, bevar1, bevar_ratio1 = PSA(brx1, n_components=6)

# Interleave the components and explained variances for plotting purposes:
comps = np.empty((bcomps0.shape[0] + bcomps1.shape[0], bcomps0.shape[1]), dtype=bcomps0.dtype)
comps[0::2, :] = bcomps0
comps[1::2, :] = bcomps1

evars = np.empty((bevar_ratio0.size + bevar_ratio0.size,), dtype=bevar_ratio0.dtype)
evars[0::2] = bevar_ratio0
evars[1::2] = bevar_ratio1


# Visualize using the mappings:
mappings = tb.open_file('/projects/francisco/data/caffe/standardized/data_mappings.h5', 'r')
baseline_mask = mappings.get_node('/r_datamask')[:]
volmask = mappings.get_node('/r_volmask')[:]
mappings.close()

baseline_shape = volmask.shape

#plt.style.use('ggplot')

sl = [(bs0, 'binary 0'), (bs1, 'binary 1'),
                               (mean_sen0, 'mean of sen 0'), (mean_sen1, 'mean of sen 1'),
                               (sen0, 'sen of mean 0'), (sen1, 'sen of mean 1')]


plot_slices(sl)

# Visualize ff1/ff2 layer activations sensitivity maps for a few of the units
plot_features(SFF1)
plot_features(SFF2)


plot_psa_slices(comps, evars)











