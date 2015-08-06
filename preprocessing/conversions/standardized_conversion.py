"""
Converts dataset extracted from ADNI to a format that is usable for python's blocks/theano neural networks.

Usage:
    standardized_conversion.py (fuel|caffe) [--scaled] [--normalized] [--targets] <target_path>

-s --scaled  scale features to [-1,1] range
-n --normalized  normalize features to mean 0, sd 1
-t --targets    change target values from [0, 1] to [-1, 1]
"""

import logging
from docopt import docopt
from fuel.datasets import H5PYDataset
import h5py
import numpy as np
import sklearn
import tables as tb
from adni_data import splits, structures, make_multi_feature_caffe_file
from preprocessing.conversions.fuel_conversions import make_one_sided_fuel_file
from adni_data import make_caffe_file


dataset = 'ad_cn'
source_path = '/projects/francisco/data/caffe/standardized/standardized_input_data_{}_{}_adcn_T2_new.h5'
source_path_labels = '/scratch/nikhil/tmp/standardized_input_classes_mcicn.h5'
target_path = '/projects/francisco/data/caffe/standardized/combined/'
features_template = '/{}_data'
fused_features_template = '/{}_data_fused'
labels_template = '/{}_classes'
fused_labels_template = '/{}_classes_fused'

sides = ['L', 'R']

X = {
    # 'EC': {
    #     'L': {},
    #     'R': {},
    #     'b': {},
    # },
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
    #source_path = arguments['<source_path>']
    logging.basicConfig(level=logging.DEBUG)
    fuel = arguments['fuel']
    caffe = arguments['caffe']

    rescale = arguments['--scaled']
    normalize = arguments['--normalized']
    scale_targets = arguments['--targets']

    # Labels:
    classes_data = tb.open_file(source_path.format('HC', 'L'), mode='r')
    #classes_data = tb.open_file(source_path_labels, 'r')
    for s in ['train', 'valid', 'test']:
        y[s] = classes_data.get_node(labels_template.format(s))[:]
        y[s + '_fused'] = classes_data.get_node(fused_labels_template.format(s))[:]
        if scale_targets:
            y[s] = (y[s] * 2) - 1.0
            y[s + '_fused'] = (y[s + '_fused'] * 2) - 1.0

    # Features:
    for side in sides:
        for structure in structures:
            data = tb.open_file(source_path.format(structure, side), mode='r')
            for s in ['train', 'valid', 'test']:
                X_c = data.get_node(features_template.format(s))[:]
                X_cf = data.get_node(fused_features_template.format(s))[:]
                if rescale:
                    X_c = (X_c * 2) - 1.0
                    X_cf = (X_cf * 2) - 1.0
                if normalize:
                    X_c = sklearn.preprocessing.normalize(X_c, norm='l1', axis=1)
                    X_cf = sklearn.preprocessing.normalize(X_cf, norm='l1', axis=1)

                X[structure][side][s] = X_c
                X[structure][side][s + '_fused'] = X_cf


            if fuel:
                num_train = X[structure][side]['train'].shape[0]
                num_valid = X[structure][side]['valid'].shape[0]
                num_test = X[structure][side]['test'].shape[0]
                X_c = np.concatenate([X[structure][side]['train'], X[structure][side]['valid'], X[structure][side]['test']], axis=0)

                y_c = np.concatenate(
                    [y['train'].reshape(-1, 1), y['valid'].reshape(-1, 1), y['test'].reshape(-1, 1)], axis=0)

                target_name = '{}{}_{}_ad_cn.h5'.format(target_path, structure, side.lower())


                make_one_sided_fuel_file(target_name, num_train, num_valid, num_test, X_c, y_c,
                               side.lower())
            data.close()

    if caffe:
        for s in ['train', 'valid', 'test']:
            target_name = '{}{}_{}_T2.h5'.format(target_path, dataset, s)
            X_c = {
                #'l_ec_features': X['EC']['L'][s],
                #'r_ec_features': X['EC']['R'][s],
                'l_hc_features': X['HC']['L'][s],
                'r_hc_features': X['HC']['R'][s],
                'l_hc_features_fused': X['HC']['L'][s + '_fused'],
                'r_hc_features_fused': X['HC']['R'][s + '_fused'],
            }
            make_multi_feature_caffe_file(target_name, X_c, y[s], y[s + '_fused'])

