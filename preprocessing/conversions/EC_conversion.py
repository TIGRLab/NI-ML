import numpy as np
import tables as tb

from preprocessing.conversions.adni_data import make_one_sided_fuel_file


rescale = False

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

    make_one_sided_fuel_file('{}{}_EC_mci_cn.h5'.format(target_path, side.lower()), num_train, num_valid, num_test, X_c, y_c, side.lower())


