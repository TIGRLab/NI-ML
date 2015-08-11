import sys
import logging
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

root = '/projects/francisco/repositories/NI-ML/'
sys.path.insert(0, root)

# Load repo-specific imports:
from adni_utils.experiment import experiment
from adni_utils.dataset_constants import *
from adni_utils.evaluate_model import evaluate


def adaboost(params, n_classes):
    learning_rate = np.exp(params['log_learning_rate'])
    n_estimators = params['n_estimators']
    max_max_depth = params['max_depth']
    # Min number of decision tree branches required to classify n_classes
    min_max_depth = int(np.ceil(np.log(n_classes)))

    max_depth = max(min_max_depth, max_max_depth)

    stump = DecisionTreeClassifier(max_depth=max_depth)

    classifier = AdaBoostClassifier(stump,
                                    algorithm="SAMME.R",
                                    n_estimators=n_estimators,
                                    learning_rate=learning_rate)
    return classifier, 'AdaBoost'


def model_metrics(classifier, var_names):
    print 'Feature Importances:'

    feats = zip(classifier.feature_importances_, var_names)
    feats.sort(reverse=True)
    for var, value in feats[0:10]:
        print '{} {}'.format(var, value)


def main(job_id, params, side=default_side, dataset=default_dataset):
    """
    Main hook for Spearmint.
    :param job_id:
    :param params:
    :return:
    """
    score = experiment(params=params, classifier_fn=adaboost, structure=structure, side=side, dataset=dataset,
                       folds=folds, source_path=source_path, use_fused=use_fused, balance=balance, n=n_trials,
                       test=False, omit_class=omit_class)
    return score


if __name__ == "__main__":
    # Entry point when running the script manually. Not run by Spearmint.
    job_id = 0
    params = {
        'log_learning_rate': -3.13853947,
        'n_estimators': 10,
        'max_depth': 3
    }
    evaluate(params=params, classifier_fn=adaboost, dataset=default_dataset,
                       source_path=source_path, n=n_trials, model_metrics=model_metrics, omit_class=omit_class)