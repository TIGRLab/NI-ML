import numpy as np
from sklearn.decomposition import RandomizedPCA
import tables as tb
from tabulate import tabulate
from PCA_Utils import transform_PCA, plot_components
from PCA_Utils import score_reconstructions
from matplotlib import pyplot as plt

#input_node = 'l_hc_features'
input_node = 'features'
label_node = 'labels'

data_path = '/projects/francisco/data/caffe/standardized/combined/balanced_ADNI_Cortical_Features_train.h5'
data_path_test = '/projects/francisco/data/caffe/standardized/combined/balanced_ADNI_Cortical_Features_test.h5'

data = tb.open_file(data_path, 'r')
train_X = data.get_node('/' + input_node)[:]
train_y = data.get_node('/' + label_node)[:]
data.close()

data = tb.open_file(data_path_test, 'r')
test_X = data.get_node('/' + input_node)[:]
test_y = data.get_node('/' + label_node)[:]
data.close()



K = []

pcas = []
K_comps = range(2,test_X.shape[1])
stats = np.zeros(shape=(len(K_comps), 3))
pca = RandomizedPCA()
pca.fit(train_X)

for i, k in enumerate(K_comps):
    print 'Fitting PCA with {} components'.format(k)
    X_reduced, X_hat_pca = transform_PCA(pca, k, test_X)
    stats[i,0] = k
    stats[i,1] = np.sum(pca.explained_variance_ratio_[:k])
    stats[i,2] = np.mean(np.mean((test_X - X_hat_pca) ** 2))

# Plot with k=2
X_reduced, X_hat = transform_PCA(pca, 2, train_X)
plot_components(X_reduced, train_y)

# plot with k=10
X_reduced, X_hat = transform_PCA(pca, 10, train_X)
plot_components(X_reduced, train_y)

print tabulate(stats, headers=['n_components', 'Var Explained', 'L2'])

plt.figure(1)
plt.xlabel('Number of Principal Components')
plt.subplot(211)
plt.title('Variance Explained Ratio')

plt.plot(K_comps, stats[:, 1], 'r')

plt.subplot(212)
plt.title('L2 Reconstruction Error')
plt.plot(K_comps, stats[:, 2], 'g')
plt.show()


