import sys
import logging
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from tabulate import tabulate

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


def model_metrics(classifiers, var_names):
    print 'Gini Importances:'

    importances = np.zeros(shape=(len(classifiers), len(var_names)))
    for i, classifier in enumerate(classifiers):
        importances[i,:] = classifier.feature_importances_


    mean_importances = np.mean(importances,axis=0)
    std_importances = np.std(importances, axis=0)
    feats = zip(var_names, mean_importances, std_importances)

    # Remove non-important feats:
    feats = [feat for feat in feats if feat[1] > 0.0]

    feats.sort(reverse=True, key=lambda x: x[1])
    print tabulate(feats, headers=['Variable', 'Mean', 'Std'])


def main(job_id, params, side=evaluation_side, dataset=evaluation_dataset):
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
        'n_estimators': 6,
        'log_learning_rate': 0.0,
        'max_depth': 1
    }
    evaluate(params=params, classifier_fn=adaboost, structure=structure, side=evaluation_side, dataset=evaluation_dataset,
                       folds=folds, source_path=source_path, use_fused=use_fused, balance=balance, n=n_trials,
                       test=False, omit_class=omit_class, model_metrics=model_metrics)
