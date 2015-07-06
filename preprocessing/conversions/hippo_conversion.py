"""
Converts dataset extracted from ADNI to a format that is usable for python's blocks/theano neural networks.

Usage:
    fuel_conversion.py (fuel|caffe) [--scaled] <target_path>

-s --scaled  scale features to [-1,1] range
"""
import logging

import numpy as np
from docopt import docopt

from adni_data import split_3_way, splits, features_template, labels_template, sides, load_data, \
    balanced_mci_indexes, input_dims
from preprocessing.conversions.adni_data import make_caffe_file, make_lr_fuel_file


if __name__ == "__main__":
    arguments = docopt(__doc__)
    target_path = arguments['<target_path>']
    rescale = arguments['--scaled']
    logging.basicConfig(level=logging.DEBUG)
    fuel = arguments['fuel']
    caffe = arguments['caffe']

    datah = load_data()
    X = {
        'l': {},
        'r': {},
        'b': {}
    }
    y = {
        'l': {},
        'r': {},
        'b': {}
    }

    hippo_dim = input_dims['HC']

    # Load all of the existing data sets:
    for side in sides:
        for s in ['train', 'valid', 'test']:
            X[side][s] = datah.get_node(features_template.format(side, s))[:]
            y[side][s] = datah.get_node(labels_template.format(side, s))[:]

    # Balance classes by removing extra MCI:
    for s in ['train', 'valid', 'test']:
        logging.info('Balancing {} Classes'.format(s))
        X[s] = {}
        y[s] = {}
        X_c = np.concatenate([X['l'][s], X['r'][s]], axis=1)
        if rescale:
            X_c = (X_c * 2) - 1.0
        y_c = y['l'][s]
        if 'train' in s:
            inds = balanced_mci_indexes(y_c)
            X_c = X_c[inds]
            y_c = y_c[inds]
        splits = split_3_way(X_c, y_c)
        for name, split in splits.items():
            X[s][name] = split['X']
            y[s][name] = split['y']


    # Reconstruct each of the 3-way classifier sets by concatenating train/valid/test sets,
    # and splitting into left and right features:
    for name, split in splits.items():
        # Concatenate training, validation, and test features/labels into single matrix:
        logging.info('Making {} Split'.format(name))
        if fuel:
            num_train = X['train'][name].shape[0]
            num_valid = X['valid'][name].shape[0]
            num_test = X['test'][name].shape[0]
            X_c = np.concatenate([X['train'][name], X['valid'][name], X['test'][name]], axis=0)
            y_c = np.concatenate(
                [y['train'][name].reshape(-1, 1), y['valid'][name].reshape(-1, 1), y['test'][name].reshape(-1, 1)], axis=0)

            # Split into left, right features again:
            X_ = {}
            X_['l'] = X_c[:, 0:hippo_dim['l']]
            X_['r'] = X_c[:, hippo_dim['l']:]

            make_lr_fuel_file('{}{}.h5'.format(target_path, name), num_train, num_valid, num_test, X_, y_c)

        if caffe:
            for s in ['train', 'valid', 'test']:
                num_s = X[s][name].shape[0]
                X_c = X[s][name]
                y_c = y[s][name]
                X_l = X_c[:, 0:hippo_dim['l']]
                X_r = X_c[:, hippo_dim['l']:]
                make_caffe_file('{}{}_{}_{}.h5'.format(target_path, 'l', name, s), X_l, y_c)
                make_caffe_file('{}{}_{}_{}.h5'.format(target_path, 'r', name, s), X_r, y_c)