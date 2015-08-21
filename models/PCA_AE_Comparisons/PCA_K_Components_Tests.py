import numpy as np
from sklearn.decomposition import RandomizedPCA
import sys
import tables as tb
from tabulate import tabulate
from PCA_Utils import transform_PCA, plot_components
from PCA_Utils import score_reconstructions
from matplotlib import pyplot as plt


ni_ml_root = '/projects/francisco/repositories/NI-ML/'

sys.path.insert(0, ni_ml_root)
from adni_utils.data import balanced_indices

input_node = 'r_hc_features'
label_node = 'labels'

data_path = '/projects/francisco/data/caffe/standardized/combined/ad_mci_cn_train.h5'
data_path_test = '/projects/francisco/data/caffe/standardized/combined/ad_mci_cn_test.h5'

data = tb.open_file(data_path, 'r')
train_X = data.get_node('/' + input_node)[:]
train_y = data.get_node('/' + label_node)[:]
train_fused_X = data.get_node('/' + input_node + '_fused')[:]
train_fused_y = data.get_node('/' + label_node + '_fused')[:]
data.close()

data = tb.open_file(data_path_test, 'r')
test_X = data.get_node('/' + input_node)[:]
test_y = data.get_node('/' + label_node)[:]
test_fused_X = data.get_node('/' + input_node + '_fused')[:]
test_fused_y = data.get_node('/' + label_node + '_fused')[:]
data.close()

# Balance classes for the 3-class data:
if 'ad_mci_cn' in data_path:
    inds = balanced_indices(train_y)
    train_X = train_X[inds]
    train_y = train_y[inds]

# Try PCA with various num_components and find the best DICE score from the resulting reconstructions.
# Keep n components, VE, dice score, l2 error, cross entropy
K = []
pcas = []
K_comps = [2**x for x in range(1,11)]
stats = np.zeros(shape=(len(K_comps), 5))

pca = RandomizedPCA(n_components=1024)
pca.fit(train_X)

for i, k in enumerate(K_comps):
    print 'Fitting PCA with {} components'.format(k)
    X_reduced, X_hat_pca = transform_PCA(pca, k, test_fused_X)
    D, L2, CC = score_reconstructions(test_fused_X, X_hat_pca)
    stats[i,0] = k
    stats[i,1] = np.sum(pca.explained_variance_ratio_[:k])
    stats[i,2] = np.mean(D)
    stats[i,3] = np.mean(np.mean((test_fused_X - X_hat_pca) ** 2))
    stats[i,4] = np.mean(CC)
    print 'Producing Plot'

print tabulate(stats, headers=['n_components', 'Var Explained', 'DICE', 'L2', 'Cross-Entropy'])

# Plot with k=2
X_reduced, X_hat = transform_PCA(pca, 2, train_fused_X)
plot_components(X_reduced, train_fused_y)

# plot with k=10
X_reduced, X_hat = transform_PCA(pca, 10, train_fused_X)
plot_components(X_reduced, train_fused_y)

plt.figure(1)
plt.xlabel('Number of Principal Components')
plt.subplot(211)
plt.title('Variance Explained Ratio')

plt.plot(K_comps, stats[:, 1], 'r')

plt.subplot(212)
plt.title('L2 Reconstruction Error')
plt.plot(K_comps, stats[:, 2], 'r')
plt.plot(K_comps, stats[:, 3], 'g')
plt.plot(K_comps, stats[:, 4], 'b')
plt.show()
