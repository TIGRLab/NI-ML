import logging
from docopt import docopt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
import sys

import tables as tb
import numpy as np
from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline

root = '/projects/francisco/repositories/NI-ML/'
sys.path.insert(0, root)

# Load repo-specific imports:
from adni_utils.experiment import experiment, test

# Data Vars:
source_path = '/projects/francisco/data/caffe/standardized/combined/'

# Sides to iterate over;
# ie: sides  = ['l', 'r', 'b']
sides = ['b']
structure = 'hc'

# Datasets to iterate over:
# ie. datasets = ['mci_cn', 'ad_cn', 'ad_mci_cn']
adni_datasets = ['ad_mci_cn']

# Use Fused only (otherwise use candidate segmentations)
use_fused = True

# Balance classes in dataset:
balance = True

# List of fold filename extensions to iterate over:
# ie. folds = ['_T1', '_T2', '_T3']
folds = [''] # No folds.

# Change these when running Spearmint experiments which use only the main and don't iterate over datasets or sides:
default_side = 'l' # Valid values: l, r, b
default_dataset = 'ad_mci_cn' # Valid values: mci_cn, ad_cn


def linearSVM(params, n_classes):
    alpha_decay = np.exp(params['log_alpha_decay'])
    lr = np.exp(params['log_learning_rate'])
    classifier = SGDClassifier(eta0=lr, alpha=alpha_decay, loss='hinge', penalty='l2', class_weight='auto')
    return classifier, 'Linear SVM'


def main(job_id, params, side=default_side, dataset=default_dataset):
    """
    Main hook for Spearmint.
    :param job_id:
    :param params:
    :return:
    """
    logging.basicConfig(level=logging.INFO)
    score = experiment(params=params, classifier_fn=linearSVM, structure=structure, side=side, dataset=dataset,
                       folds=folds, source_path=source_path, use_fused=use_fused, balance=balance)
    return score


if __name__ == "__main__":
    #arguments = docopt(__doc__)
    held_out_test = True
    job_id = 0
    params = {
        'log_alpha_decay': -10.0,
        'log_learning_rate': -10.0
    }
    if held_out_test:
        test(params=params, classifier_fn=linearSVM, structure=structure, side=default_side, dataset=default_dataset,
                    source_path=source_path, use_fused=use_fused, balance=balance)
    else:
        for side in sides:
            for dataset in adni_datasets:
                main(job_id, params, side, dataset)




