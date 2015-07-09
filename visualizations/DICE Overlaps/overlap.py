'''
Script to compute various DICE scores on:
a) Pairwise DICE on features in class 0 vs features in class 1
b) DICE distance matrix on features in both classes separately
'''

import logging
import os
import tables as tb
import numpy as np
from matplotlib.pyplot import savefig, cla
from scipy.spatial.distance import cdist, pdist
from pylab import pcolor, show, colorbar, xticks, yticks

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

def report(D, score):
    """
    Just prints stuff.
    :param D:
    :return:
    """
    print 'Max {} (least similar): {}'.format(score, np.max(D))
    print 'Mean {} (mean similarity): {}'.format(score, np.mean(D))
    print 'Min {} (most similar): {}'.format(score, np.min(D))


def compute_dice_scores(features, inds0, inds1):
    """
    Compute and report dem scores.
    :param features:
    :param inds0:
    :param inds1:
    :return:
    """
    g0 = features[inds0,:]
    g1 = features[inds1,:]
    D = cdist(g0, g1, 'dice')
    report(D, '0 vs 1 DICE')
    D0=pdist(g0, 'dice')
    report(D0, '0 distance')
    D1=pdist(g1, 'dice')
    report(D1, '1 distance')
    return D, D0, D1


source_path = '/scratch/nikhil/tmp/standardized_input_data_{}_{}_mini.h5'
source_path_labels = '/scratch/nikhil/tmp/standardized_input_classes_mini.h5'

labels_data = tb.open_file(source_path_labels, mode='r')

labels = labels_data.get_node('/valid_classes')[:]
inds0 = np.where(labels == 0)[0]
inds1 = np.where(labels == 1)[0]

scores = {}

for side in ['L', 'R']:
    for structure in ['HC', 'EC']:
        print 'Computing DICE overlaps for {}_{}:'.format(side, structure)
        print 'This might take a minute...'
        data = tb.open_file(source_path.format(structure, side), 'r')
        features = data.get_node('/valid_data')[:]
        D, D0, D1 = compute_dice_scores(features, inds0, inds1)
        scores['{}_{}'.format(side, structure)] = (D, D0, D1)







