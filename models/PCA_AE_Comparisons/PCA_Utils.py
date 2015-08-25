import numpy as np
from sklearn.decomposition import RandomizedPCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import dice
from sklearn.metrics import log_loss
from matplotlib import pyplot as plt
from sklearn.utils.extmath import fast_dot


def plot_components(X_hat, y):
    if X_hat.shape[1] > 2:
        model = TSNE(n_components=2)
        X = model.fit_transform(X_hat)
    else:
        X = X_hat

    plt.cla()
    ind0 = np.where(y == 0)[0]
    ind1 = np.where(y == 1)[0]
    plt.scatter(X[ind0, 0], X[ind0, 1], c='mediumturquoise', alpha=0.5)
    plt.scatter(X[ind1, 0], X[ind1, 1], c='slategray', alpha=0.5)
    plt.title('{} component PCA'.format(X_hat.shape[1]))
    if np.unique(y).shape[0] > 2:
        ind2 = np.where(y == 2)[0]
        plt.scatter(X[ind2, 0], X[ind2, 1], c='red', alpha=0.5)
    plt.show()



def transform_PCA(pca, k, X):
    X_reduced = X - pca.mean_

    X_reduced = fast_dot(X_reduced, pca.components_[0:k].T)

    # Transform test data with principal components:
    #X_reduced = pca.transform(test_X)

    # Reconstruct:
    X_rec = np.dot(X_reduced, pca.components_[0:k])

    # Restore mean:
    X_rec += pca.mean_
    print "Variance Explained: {}".format(np.sum(pca.explained_variance_ratio_[:k]))
    return X_reduced, X_rec


def score_reconstructions(X, X_hat):
    """
    Score the reconstructions using DICE, l2 error, and cross entropy
    :param X:
    :param X_hat:
    :return:
    """
    D = []
    L2 = []
    CC = []
    for i in range(X.shape[0]):
        try:
            dice_score = dice(X[i].astype(int), np.round_(X_hat[i], 0).astype(int))
        except ZeroDivisionError:
            dice_score = 0.0
        D.append(dice_score)

        L2.append(np.mean((X - X_hat) ** 2))

        CC.append(log_loss(X, X_hat))

    print 'Mean DICE Dissimilarity Score (0.0 is no dissimilarity, 1.0 is total dissimilarity): {} '.format(np.mean(D))
    return D, L2, CC