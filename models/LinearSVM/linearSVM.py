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


def linearSVM(params, n_classes):
    alpha_decay = np.exp(params['log_alpha_decay'])
    lr = np.exp(params['log_learning_rate'])
    classifier = SGDClassifier(eta0=lr, alpha=alpha_decay, loss='hinge', penalty='l2', class_weight='auto', n_iter=200)
    return classifier, 'Linear SVM'


def main(job_id, params):
    """
    Main hook for Spearmint.
    :param job_id:
    :param params:
    :return:
    """
    score = experiment(params=params, classifier_fn=linearSVM, n=default_n_trials, test=False, **dataset_args[default_dataset])
    return score


if __name__ == "__main__":
    #arguments = docopt(__doc__)
    params = {
        'log_learning_rate': -0.80841064,
        'log_alpha_decay': -13.08461304,
    }
    evaluate(params=params, classifier_fn=linearSVM, n=default_n_trials, test=False, model_metrics=None, **dataset_args[default_dataset])





