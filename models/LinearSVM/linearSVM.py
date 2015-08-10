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
from adni_utils.experiment import experiment
from adni_utils.dataset_constants import *


def linearSVM(params, n_classes):
    alpha_decay = np.exp(params['log_alpha_decay'])
    lr = np.exp(params['log_learning_rate'])
    classifier = SGDClassifier(eta0=lr, alpha=alpha_decay, loss='hinge', penalty='l2', class_weight='auto', n_iter=20)
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
                       folds=folds, source_path=source_path, use_fused=use_fused, balance=balance, n=n_trials)
    return score


if __name__ == "__main__":
    #arguments = docopt(__doc__)
    held_out_test = True
    job_id = 0
    params = {
        'log_alpha_decay': -1.69406367,
        'log_learning_rate': -1.69406367
    }
    if held_out_test:
        experiment(params=params, classifier_fn=linearSVM, structure=structure, side=default_side, dataset=default_dataset,
                    folds=folds, source_path=source_path, use_fused=use_fused, balance=balance, n=n_trials, test=True)
    else:
        for side in sides:
            for dataset in adni_datasets:
                main(job_id, params, side, dataset)




