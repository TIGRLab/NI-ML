import logging
from fuel.datasets import H5PYDataset
import h5py
import numpy as np
import tables as tb
from adni_data import splits, hippo_dim, structures

rescale = False


def make_fuel_file(outfile, inda, indb, indc, X, y, side):
    # Make the pytables table:
    f = h5py.File(outfile, mode='w')
    targets = f.create_dataset('targets', y.shape, dtype='int8')
    features = f.create_dataset('{}_features'.format(side), X.shape, dtype='int8')

    # Load the data into it:
    features[...] = X
    targets[...] = y

    # Label the axis:
    targets.dims[0].label = 'sample'
    targets.dims[1].label = 'class'
    features.dims[0].label = 'sample'
    features.dims[1].label = 'feature'

    # Make a "splits" dictionary as required by Fuel
    split_dict = {
        'train': {'{}_features'.format(side): (0, inda),
                  'targets': (0, inda)},
        'valid': {'{}_features'.format(side): (inda, inda + indb),
                  'targets': (inda, inda + indb)},
        'test': {'{}_features'.format(side): (inda + indb, inda + indb + indc),
                 'targets': (inda + indb, inda + indb + indc)},
    }

    f.attrs['split'] = H5PYDataset.create_split_array(split_dict)

    # Save this new dataset to file
    f.flush()
    f.close()


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

        make_fuel_file('{}{}_{}_ad_cn.h5'.format(target_path, structure, side.lower()), num_train, num_valid, num_test, X_c, y_c,
                       side.lower())



