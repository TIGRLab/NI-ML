import logging
from docopt import docopt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

import tables as tb
import numpy as np
from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline

# Data Vars:
source_path = '/projects/francisco/data/caffe/standardized/combined/'
model = 'Linear SVM'

# Sides to iterate over;
# ie: sides  = ['l', 'r', 'b']
sides = ['b']
structure = 'hc'

# Datasets to iterate over:
# ie.
# datasets = ['mci_cn', 'ad_cn']
datasets = ['mci_cn']

# List of fold filename extensions to iterate over:
# ie:
# folds = ['_T1', '_T2', '_T3']
folds = [''] # No folds.

# Change these when running Spearmint experiments which use only the main and don't iterate over datasets or sides:
default_side = 'l' # Valid values: l, r, b
default_dataset = 'mci_cn' # Valid values: mci_cn, ad_cn


def linearSVM(C):
    return LinearSVC(C=C)


def load_dataset(fold, side, dataset):
    train_data_file = '{}_{}{}.h5'.format(dataset, 'train', fold)
    valid_data_file = '{}_{}{}.h5'.format(dataset, 'valid', fold)
    test_data_file = '{}_{}{}.h5'.format(dataset, 'test', fold)

    source_path = '/projects/francisco/data/caffe/standardized/combined/'

    d = tb.open_file(source_path + train_data_file)
    d_valid = tb.open_file(source_path + valid_data_file)
    d_test = tb.open_file(source_path + test_data_file)

    if 'b' in side:
        X_l = d.get_node('/l_{}_features_fused'.format(structure.lower()))[:]
        X_r = d.get_node('/r_{}_features_fused'.format(structure.lower()))[:]
        X_vl = d_valid.get_node('/l_{}_features_fused'.format(structure.lower()))[:]
        X_vr = d_valid.get_node('/r_{}_features_fused'.format(structure.lower()))[:]
        X_tl = d_test.get_node('/l_{}_features_fused'.format(structure.lower()))[:]
        X_tr = d_test.get_node('/r_{}_features_fused'.format(structure.lower()))[:]

        X = np.concatenate([X_l, X_r], axis=1)
        X_v = np.concatenate([X_vl, X_vr], axis=1)
        X_t = np.concatenate([X_tl, X_tr], axis=1)
    else:
        X = d.get_node('/{}_{}_features_fused'.format(side, structure.lower()))[:]
        X_v = d_valid.get_node('/{}_{}_features_fused'.format(side, structure.lower()))[:]
        X_t = d_test.get_node('/{}_{}_features_fused'.format(side, structure.lower()))[:]

    X = normalize(X)
    X_v = normalize(X_v)
    X_t = normalize(X_t)

    y = d.get_node('/label_fused')[:]
    y_v = d_valid.get_node('/label_fused')[:]
    y_t = d_test.get_node('/label_fused')[:]

    d.close()
    d_valid.close()
    d_test.close()

    return X, X_v, X_t, y, y_v, y_t


def experiment_on_fold(params, X, X_v, X_t, y, y_v, y_t):
    C = params['C']
    classifier = linearSVM(C)

    logging.info('Fitting {}'.format(model))
    classifier.fit(X, y)

    logging.info('Validating {}'.format(model))
    vscore = classifier.score(X_v, y_v)

    logging.info('Testing {}'.format(model))
    tscore = classifier.score(X_t, y_t)

    return vscore, tscore



def main(job_id, params, side=default_side, dataset=default_dataset):
    """

    :param job_id:
    :param params:
    :return:
    """
    logging.basicConfig(level=logging.INFO)
    print 'Running Experiment on side {} of dataset {}:'.format(side, dataset)
    print 'Using Parameters: '
    print params
    total_vscore = 0.0
    total_tscore = 0.0


    for i, fold in enumerate(folds):
        print "Fold {}:".format(i)
        X, X_v, X_t, y, y_v, y_t = load_dataset(fold, side, dataset)
        vscore, tscore = experiment_on_fold(params, X, X_v, X_t, y, y_v, y_t)
        print 'Validation Score: {}'.format(vscore)
        print 'Test Score: {}'.format(tscore)
        print
        total_tscore += tscore
        total_vscore += vscore


    total_tscore /= len(folds)
    total_vscore /= len(folds)

    print 'Avg Validation Score: {}'.format(total_vscore)
    print 'Avg Test Score: {}'.format(total_tscore)
    print


    # Minimize error (for spearmint):
    return (1.0 - total_vscore)

if __name__ == "__main__":
    #arguments = docopt(__doc__)
    job_id = 0
    arguments = {
        'C': 0.5,
    }
    for side in sides:
        for dataset in datasplits:
            main(job_id, arguments, side, dataset)




