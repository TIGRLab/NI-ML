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
from adni_utils.evaluate_model import evaluate

root = '/projects/francisco/repositories/NI-ML/'
sys.path.insert(0, root)

# Load repo-specific imports:
from adni_utils.experiment import experiment
from adni_utils.dataset_constants import *


def rbfSVM(params, n_classes):
    C = np.exp(params['log_C'])
    gamma = np.exp(params['log_gamma'])
    classifier = SVC(C=C, gamma=gamma, kernel='rbf')
    return classifier, 'RBF Kernel SVM'


def main(job_id, params, side=evaluation_side, dataset=evaluation_dataset):
    """
    Main hook for Spearmint.
    :param job_id:
    :param params:
    :return:
    """
    score = experiment(params=params, classifier_fn=rbfSVM, structure=structure, side=side, dataset=dataset,
                       folds=folds, source_path=source_path, use_fused=use_fused, balance=balance, n=n_trials)
    return score


if __name__ == "__main__":
    #arguments = docopt(__doc__)
    held_out_test = True
    job_id = 0
    params = {
        'log_gamma': -1.69406367,
        'log_C': 1
    }
    evaluate(params=params, classifier_fn=rbfSVM, dataset=evaluation_dataset,
                       source_path=source_path, n=n_trials)




