import numpy as np
from scipy.spatial.distance import dice
from adni_data import load_data

data = load_data()

l_features = data.get_node('/l_train_data')[:]
r_features = data.get_node('/r_train_data')[:]
labels = data.get_node('/r_train_classes')[:]

features = np.concatenate((l_features, r_features), axis=1)

# indexes:
mci_inds = np.where(labels == 2)[0]
cn_inds = np.where(labels == 1)[0]

mci_features = features[mci_inds, :]
cn_features = features[cn_inds, :]

scores = {}

for i, mci in enumerate(mci_features):
    for j, cn in enumerate(cn_features):
        score = dice(mci, cn)
        scores[(i, j)] = score







