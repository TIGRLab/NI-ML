import numpy as np
from scipy.spatial.distance import cdist
from adni_data import load_data
from pylab import pcolor, show, colorbar, xticks, yticks

def dice_plot(R):
    pcolor(R)
    colorbar()
    yticks(np.arange(0.5,10.5),range(0,10))
    xticks(np.arange(0.5,10.5),range(0,10))
    show()

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

D = cdist(mci_features, cn_features, 'dice')







