"""
Converts dataset extracted from ADNI to a format that is usable for python's blocks/theano neural networks.

Usage:
    fuel_conversion.py (fuel|caffe) <target_path>
"""
import logging
import numpy as np
import h5py
from docopt import docopt
from fuel.datasets.hdf5 import H5PYDataset
from adni_data import split_3_way, splits, features_template, labels_template, sides, datafile, load_data, \
    balanced_mci_indexes, input_dim


def make_fuel_file(outfile, inda, indb, indc, X, y):
    # Make the pytables table:
    f = h5py.File(outfile, mode='w')
    targets = f.create_dataset('targets', y.shape, dtype='int8')
    l_features = f.create_dataset('l_features', X['l'].shape, dtype='int8')
    r_features = f.create_dataset('r_features', X['r'].shape, dtype='int8')

    # Load the data into it:
    l_features[...] = X['l']
    r_features[...] = X['r']
    targets[...] = y

    # Label the axis:
    targets.dims[0].label = 'sample'
    targets.dims[1].label = 'class'
    l_features.dims[0].label = 'sample'
    l_features.dims[1].label = 'feature'
    r_features.dims[0].label = 'sample'
    r_features.dims[1].label = 'feature'

    # Make a "splits" dictionary as required by Fuel
    split_dict = {
        'train': {'l_features': (0, inda),
                  'r_features': (0, inda),
                  'targets': (0, inda)},
        'valid': {'l_features': (inda, inda + indb),
                  'r_features': (inda, inda + indb),
                  'targets': (inda, inda + indb)},
        'test': {'l_features': (inda + indb, inda + indb + indc),
                 'r_features': (inda + indb, inda + indb + indc),
                 'targets': (inda + indb, inda + indb + indc)},
    }

    f.attrs['split'] = H5PYDataset.create_split_array(split_dict)

    # Save this new dataset to file
    f.flush()
    f.close()

def make_caffe_file(outfile, X, y):
    """
    Create a Caffe-format HDf5 data and labels file.
    :param outfile: A path and filename to write the dataset out to.
    :param set_name: Name of the dataset (ie. 'left' or 'right')
    :param X: The features matrix.
    :param y: The class labels vector.
    """
    # Make the pytables table:
    f = h5py.File(outfile, mode='w')
    label = f.create_dataset('label', y.shape)
    set_name = f.create_dataset('features', X.shape)

    # Load the data into it:
    set_name[...] = X
    label[...] = y

    # Save this new dataset to file
    f.flush()
    f.close()


if __name__ == "__main__":
    arguments = docopt(__doc__)
    target_path = arguments['<target_path>']
    format = arguments['<format>']
    logging.basicConfig(level=logging.DEBUG)

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
        if 'fuel' in format:
            num_train = X['train'][name].shape[0]
            num_valid = X['valid'][name].shape[0]
            num_test = X['test'][name].shape[0]
            X_c = np.concatenate([X['train'][name], X['valid'][name], X['test'][name]], axis=0)
            y_c = np.concatenate(
                [y['train'][name].reshape(-1, 1), y['valid'][name].reshape(-1, 1), y['test'][name].reshape(-1, 1)], axis=0)

            # Split into left, right features again:
            X_ = {}
            X_['l'] = X_c[:, 0:input_dim['l']]
            X_['r'] = X_c[:, input_dim['l']:]

            make_fuel_file('{}{}.h5'.format(target_path, name), num_train, num_valid, num_test, X_, y_c)

        elif 'caffe' in format:
            for s in ['train', 'valid', 'test']:
                num_s = X[s][name].shape[0]
                X_c = X[s][name]
                y_c = y[s][name]
                X_l = X_c[:, 0:input_dim['l']]
                X_r = X_c[:, input_dim['l']:]
                make_caffe_file('{}{}_{}_{}.h5'.format(target_path, 'l', name, s), X_l, y_c)
                make_caffe_file('{}{}_{}_{}.h5'.format(target_path, 'r', name, s), X_r, y_c)