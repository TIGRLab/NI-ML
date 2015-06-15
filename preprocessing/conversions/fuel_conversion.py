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
from adni_data import split_3_way, splits, features_template, labels_template, sides, datafile, load_data


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
    X = {}

    for k in splits.keys():
        for side in sides:
            logging.info('Making {} Split...'.format(k))
            train = datah.get_node(features_template.format(side, 'train'))[:]
            train_labels = datah.get_node(labels_template.format(side, 'train'))[:]
            train_splits = split_3_way(train, train_labels)

            valid = datah.get_node(features_template.format(side, 'valid'))[:]
            valid_labels = datah.get_node(labels_template.format(side, 'valid'))[:]
            valid_splits = split_3_way(valid, valid_labels)

            test = datah.get_node(features_template.format(side, 'test'))[:]
            test_labels = datah.get_node(labels_template.format(side, 'test'))[:]
            test_splits = split_3_way(test, test_labels)

            train_X = train_splits[k]['X']
            valid_X = valid_splits[k]['X']
            test_X = test_splits[k]['X']
            train_y = train_splits[k]['y']
            valid_y = valid_splits[k]['y']
            test_y = test_splits[k]['y']

            # Concatenate training, validation, and test features/labels into single matrix:
            X_c = np.concatenate([train_X, valid_X, test_X], axis=0)
            y_c = np.concatenate(
                [train_y.reshape(-1, 1), valid_y.reshape(-1, 1), test_y.reshape(-1, 1)], axis=0)
            X[side] = X_c

        make_file('{}{}.h5'.format(target_path, k), train_X.shape[0], valid_X.shape[0], test_X.shape[0], X, y_c)