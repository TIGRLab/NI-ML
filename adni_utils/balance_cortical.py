import numpy as np
import tables as tb
import sys

root = '/projects/francisco/repositories/NI-ML/'
sys.path.insert(0, root)

from adni_utils.data import balanced_indices
from preprocessing.conversions.adni_data import make_caffe_file

target = '/projects/francisco/data/caffe/standardized/combined/'
source = '/projects/francisco/data/caffe/standardized/combined/'
splits = ['test', 'train', 'valid']
dataset = 'ADNI_Cortical_Features'

for split in splits:
    file_tmp = '{}_{}'.format(dataset, split)
    source_file = source + file_tmp + '.h5'
    target_file = target + 'balanced_' + file_tmp + '.h5'
    data = tb.open_file(source_file, 'r')

    # Balanced data temp storage:
    y = data.get_node('/labels')[:]
    inds = balanced_indices(y)
    bal_y = y[inds]
    node = 'features'
    X = data.get_node('/' + node)
    balanced_X = X[inds,:]

    data.close()
    make_caffe_file(target_file, balanced_X, bal_y)

