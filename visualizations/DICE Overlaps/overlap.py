'''
Run this script from the NI-ML repository root folder so it finds the adni_data dependency.

Last run on *test* sets produced:
mci_cn:
Max Dice (least similar): 0.586656322731
Min Dice (most similar): 0.165756630265

ad_cn:
Max Dice (least similar): 0.581625604421
Min Dice (most similar): 0.163028021748

ad_mci:
Max Dice (least similar): 0.585340037047
Min Dice (most similar): 0.162198649952
'''

import logging
import os
from matplotlib.pyplot import savefig, cla
import numpy as np
from scipy.spatial.distance import cdist
from pylab import pcolor, show, colorbar, xticks, yticks
from preprocessing.conversions.adni_data import load_data, class_labels, splits

target_folder = './visualizations/DICE Overlaps'


def dice_plot(M, name):
    """
    Produces a DICE score matrix plot for the given matrix.
    :param M:
    :return:
    """
    cla()
    pcolor(M)
    colorbar()
    yticks(np.arange(0.5, 10.5), range(0, 10))
    xticks(np.arange(0.5, 10.5), range(0, 10))

    savefig('{}.png'.format(name))


data = load_data()

l_features = data.get_node('/l_test_data')[:]
r_features = data.get_node('/r_test_data')[:]
labels = data.get_node('/r_test_classes')[:]

features = np.concatenate((l_features, r_features), axis=1)

sets = {}

for group, label in class_labels.items():
    inds = np.where(labels == label)[0]
    sets[group] = features[inds, :]

for split in splits.keys():
    logging.info('Scoring DICE for {} sets'.format(split))
    g1, g2 = split.split('_')
    g1_features = sets[g1]
    g2_features = sets[g2]
    D = cdist(g1_features, g2_features, 'dice')
    print 'Max Dice (least similar): {}'.format(np.max(D))
    print 'Min Dice (most similar): {}'.format(np.min(D))
    dice_plot(D, split)








