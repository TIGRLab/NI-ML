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
from adni_utils.evaluate_model import evaluate



def rbfSVM(params, n_classes):
    C = np.exp(params['log_C'])
    gamma = np.exp(params['log_gamma'])
    classifier = SVC(C=C, gamma=gamma, kernel='rbf', probability=True)
    return classifier, 'RBF Kernel SVM'


def main(job_id, params):
    """
    Main hook for Spearmint.
    :param job_id:
    :param params:
    :return:
    """
    score = experiment(params=params, classifier_fn=rbfSVM, n=default_n_trials, test=False, **dataset_args[default_dataset])

    return score


if __name__ == "__main__":
    #arguments = docopt(__doc__)
    held_out_test = True
    job_id = 0
    params = {
        'log_gamma': -1.69406367,
        'log_C': 1
    }
    evaluate(params=params, classifier_fn=rbfSVM, n=default_n_trials, test=False, model_metrics=None, **dataset_args[default_dataset])





