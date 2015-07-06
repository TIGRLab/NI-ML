import logging
from fuel.datasets import H5PYDataset
import h5py
import numpy as np
import tables as tb
from hippo_conversion import make_fuel_file
from adni_data import splits, hippo_dim

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

source_path = '/projects/nikhil/miccai/input_data_comb/EC_input_dataset_mcicn_{}.h5'
target_path = '/projects/francisco/data/fuel/EC/'
features_template = '/{}_data'
labels_template = '/{}_classes'

sides = ['L', 'R']

X = {
    'L': {},
    'R': {},
    'b': {}
}
y = {
    'L': {},
    'R': {},
    'b': {}
}

# Load all of the existing data sets:
for side in sides:
    data = tb.open_file(source_path.format(side), mode='r')
    for s in ['train', 'valid', 'test']:
        X[side][s] = data.get_node(features_template.format(s))[:]
        y[side][s] = data.get_node(labels_template.format(s))[:]

    num_train = X[side]['train'].shape[0]
    num_valid = X[side]['valid'].shape[0]
    num_test = X[side]['test'].shape[0]
    X_c = np.concatenate([X[side]['train'], X[side]['valid'], X[side]['test']], axis=0)
    if rescale:
            X_c = (X_c * 2) - 1.0
    y_c = np.concatenate(
        [y[side]['train'].reshape(-1, 1), y[side]['valid'].reshape(-1, 1), y[side]['test'].reshape(-1, 1)], axis=0)

    make_fuel_file('{}{}_EC_mci_cn.h5'.format(target_path, side.lower()), num_train, num_valid, num_test, X_c, y_c, side.lower())


