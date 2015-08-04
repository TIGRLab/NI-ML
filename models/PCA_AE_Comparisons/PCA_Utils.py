import numpy as np
from sklearn.decomposition import RandomizedPCA
from scipy.spatial.distance import dice

def transform_PCA(k, train_X, test_X):
    pca = RandomizedPCA(n_components=k)
    pca.fit(train_X)

    # Transform test data with principal components:
    X_reduced = pca.transform(test_X)

    # Reconstruct:
    X_rec = np.dot(X_reduced, pca.components_)

    # Restore mean:
    X_rec += pca.mean_
    print "Variance Explained: {}".format(pca.explained_variance_ratio_)
    return pca, X_rec


def score_reconstructions(X, X_hat):
    D = []
    for i in range(X.shape[0]):
        score = dice(X[i].astype(int), np.round_(X_hat[i], 0).astype(int))
        D.append(score)
    print 'Mean DICE Dissimilarity Score (0.0 is no dissimilarity, 1.0 is total dissimilarity): {} '.format(np.mean(D))
    return D