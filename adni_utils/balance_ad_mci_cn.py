import numpy as np
import tables as tb
import sys

root = '/projects/francisco/repositories/NI-ML/'
sys.path.insert(0, root)

from adni_utils.data import balanced_indices
from preprocessing.conversions.adni_data import make_multi_feature_caffe_file

target = '/projects/francisco/data/caffe/standardized/combined/'
source = '/projects/francisco/data/caffe/standardized/combined/'
splits = ['test', 'train', 'valid']
dataset = 'ad_mci_cn'

for split in splits:
    file_tmp = '{}_{}'.format(dataset, split)
    source_file = source + file_tmp + '.h5'
    target_file = target + 'balanced_' + file_tmp + '.h5'
    data = tb.open_file(source_file, 'r')

    # Balanced data temp storage:
    X_all = {}

    # For fused:
    y_fused = data.get_node('/labels_fused')[:]
    inds = balanced_indices(y_fused)
    y_fused = y_fused[inds]
    for node in ['r_hc_features_fused', 'l_hc_features_fused']:
        X = data.get_node('/' + node)
        balanced_X = X[inds,:]
        X_all[node] = balanced_X.copy()

    y = data.get_node('/labels')[:]
    inds = balanced_indices(y)
    y = y[inds]
    for node in ['r_hc_features', 'l_hc_features']:
        X = data.get_node('/' + node)
        balanced_X = X[inds,:]
        X_all[node] = balanced_X.copy()

    data.close()
    make_multi_feature_caffe_file(target_file, X_all, y, y_fused)


