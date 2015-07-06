"""
Converts dataset extracted from ADNI to a format that is usable for python's blocks/theano neural networks.

Usage:
    fuel_conversion.py (fuel|caffe) [--scaled] <target_path>

-s --scaled  scale features to [-1,1] range
"""

import logging
from docopt import docopt
from fuel.datasets import H5PYDataset
import h5py
import numpy as np
import tables as tb
from adni_data import splits, structures, make_one_sided_fuel_file
from preprocessing.conversions.adni_data import make_caffe_file


source_path = '/scratch/nikhil/tmp/standardized_input_data_{}_{}_mini.h5'
source_path_labels = '/scratch/nikhil/tmp/standardized_input_classes_mini.h5'
target_path = '/projects/francisco/data/fuel/standardized/'
features_template = '/{}_data'
labels_template = '/{}_classes'

sides = ['L', 'R']

X = {
    'EC': {
        'L': {},
        'R': {},
        'b': {},
    },
    'HC': {
        'L': {},
        'R': {},
        'b': {},
    },
}

y = {}

if __name__ == "__main__":
    arguments = docopt(__doc__)
    target_path = arguments['<target_path>']
    rescale = arguments['--scaled']
    logging.basicConfig(level=logging.DEBUG)
    fuel = arguments['fuel']
    caffe = arguments['caffe']

    # Labels:
    classes_data = tb.open_file(source_path_labels, mode='r')
    for s in ['train', 'valid', 'test']:
        y[s] = classes_data.get_node(labels_template.format(s))[:]

    # Features:
    for side in sides:
        for structure in structures:
            data = tb.open_file(source_path.format(structure, side), mode='r')
            for s in ['train', 'valid', 'test']:
                X[structure][side][s] = data.get_node(features_template.format(s))[:]

            num_train = X[structure][side]['train'].shape[0]
            num_valid = X[structure][side]['valid'].shape[0]
            num_test = X[structure][side]['test'].shape[0]
            X_c = np.concatenate([X[structure][side]['train'], X[structure][side]['valid'], X[structure][side]['test']], axis=0)

            if rescale:
                X_c = (X_c * 2) - 1.0

            y_c = np.concatenate(
                [y['train'].reshape(-1, 1), y['valid'].reshape(-1, 1), y['test'].reshape(-1, 1)], axis=0)

            target_name = '{}{}_{}_ad_cn.h5'.format(target_path, structure, side.lower())

            if fuel:
                make_one_sided_fuel_file(target_name, num_train, num_valid, num_test, X_c, y_c,
                           side.lower())
            if caffe:
                make_caffe_file(target_name, X_c, y_c)



