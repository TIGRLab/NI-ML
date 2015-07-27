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
from sensitivity import sample_Sjk, normed_sen, sample_binary_perturbation_Sjk, sample_binary_FF_Sjk, PSA, plot_slices, \
    plot_features, plot_psa_slices, Sjk

caffe_root = '/home/fran/workspace/caffe/'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')
import caffe

# GPU mode
caffe.set_device(0)
caffe.set_mode_gpu()
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
H0 = np.identity(X.shape[1]) * sensitivity_factor
H1 = np.identity(X.shape[1]) * sensitivity_factor

BATCH = H0.shape[0]

# Sensitivity for kth input for class j is (recall F() is the forward pass of the nn):
# (F(xmu_j + H(j,k)) - F(xmu_j - H(j,k)) / 2h_j
# By limit laws, the above gives dF(xmu_j) / dx_j
outp, outn, DX0 = Sjk(net, mu_0, H0)
s0k = DX0[:, 0]

outp1, outn1, DX1 = Sjk(net, mu_1, H1)
s1k = DX1[:, 0]

sen0, sen1 = normed_sen(s0k, s1k)

# Mean Class Sensitivity using test data:
find0 = np.where(y_fused == 0)[0]
find1 = np.where(y_fused == 1)[0]

X_fused_0 = X_fused[find0, :]
X_fused_1 = X_fused[find1, :]

DX0 = sample_Sjk(net, X_fused_0, sensitivity_factor)
sx0 = np.mean(DX0[:, :, 0], axis=0)
# X_fused_0_mu = np.mean(X_fused_0, axis=0)
# sx0 = DX0[:, :, 0] / (2 * X_fused_0_mu * sensitivity_factor)

DX1 = sample_Sjk(net, X_fused_1, sensitivity_factor)
sx1 = np.mean(DX1[:, :, 0], axis=0)
# sx1 = DX1[:, :, 0] / (2 * np.mean(X_fused_1, axis=0) * sensitivity_factor)

mean_sen0, mean_sen1 = normed_sen(sx0, sx1)

# Binary sensitivity over the test set for each class:
# Note: should we only go over the X_fused data for that class?
# If so, then we possibly miss out on learning about counter-examples.
bDX = sample_binary_perturbation_Sjk(net, X_fused)
brx0 = bDX[:, :, 0]
brx1 = bDX[:, :, 0]
bs0 = np.mean(brx0, axis=0)
bs1 = np.mean(brx1, axis=0)

normed_bs0, normed_bs1 = normed_sen(bs0, bs1)


# FF1 and FF2 activations:
FF1, FF2 = sample_binary_FF_Sjk(net, X_fused)
FF1mu = np.mean(FF1, axis=0).T
FF2mu = np.mean(FF2, axis=0).T

# Principal Sensitivity Analysis
# K = E[r(x)r(x).T] where r(x) is a k-dimensional vector of the partial derivatives of F(x) w/ respect to x_k
# Note that E[r(x)] is already computed above for each class, taking the expectation over the test data.

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

# plt.style.use('ggplot')

sl = [(bs0, 'binary 0'), (bs1, 'binary 1'),
      (mean_sen0, 'mean of sen 0'), (mean_sen1, 'mean of sen 1'),
      (sen0, 'sen of mean 0'), (sen1, 'sen of mean 1')]

#sla = [(sx0, 'sx0'), (sx1, 'sx1'), (s0k, 's0k'), (s1k, 's1k')]

plot_slices(sl, baseline_shape, baseline_mask)

# Visualize ff1/ff2 layer activations sensitivity maps for a few of the units
plot_features(FF1mu, baseline_shape, baseline_mask)
plot_features(FF2mu, baseline_shape, baseline_mask)

plot_psa_slices(comps, evars, baseline_shape, baseline_mask)












