"""
Converts dataset extracted from ADNI to a format that is usable for python's blocks/theano neural networks.

Usage:
    fuel_conversion.py <target_path>
"""
import logging
import numpy as np
import h5py
from docopt import docopt
from fuel.datasets.hdf5 import H5PYDataset
from adni_data import split_3_way, splits, features_template, labels_template, sides, datafile, load_data, \
    balanced_mci_indexes, input_dim


def make_file(outfile, inda, indb, indc, X, y):
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


if __name__ == "__main__":
    arguments = docopt(__doc__)
    target_path = arguments['<target_path>']
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

    for side in sides:
        for s in ['train', 'valid', 'test']:
            X[side][s] = datah.get_node(features_template.format(side, s))[:]
            y[side][s] = datah.get_node(labels_template.format(side, s))[:]

    # Balance MCI classes:
    for s in ['train', 'valid', 'test']:
        logging.info('Balancing {} Classes'.format(s))
        X[s] = {}
        y[s] = {}
        X_c = np.concatenate([X['l'][s], X['r'][s]], axis=1)
        y_c = y['l'][s]
        inds = balanced_mci_indexes(y_c)
        X_c = X_c[inds]
        y_c = y_c[inds]
        for key, split in split_3_way(X_c, y_c).items():
            X[s][key] = split['X']
            y[s][key] = split['y']


    for key, split in splits.items():
        # Concatenate training, validation, and test features/labels into single matrix:
        logging.info('Making {} Split'.format(key))
        num_train = X['train'][key].shape[0]
        num_valid = X['valid'][key].shape[0]
        num_test = X['test'][key].shape[0]
        X_c = np.concatenate([X['train'][key], X['valid'][key], X['test'][key]], axis=0)
        y_c = np.concatenate(
            [y['train'][key].reshape(-1, 1), y['valid'][key].reshape(-1, 1), y['test'][key].reshape(-1, 1)], axis=0)

        X_l = X_c[:, 0:input_dim['l']]
        X_r = X_c[:, input_dim['l']:]

        X_ = {}
        X_['l'] = X_l
        X_['r'] = X_r

        make_file('{}{}.h5'.format(target_path, key), num_train, num_valid, num_test, X_, y_c)