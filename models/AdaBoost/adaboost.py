import sys
import logging
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from tabulate import tabulate
from matplotlib import pyplot as plt

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
    importances_std = np.zeros(shape=(len(classifiers), len(var_names)))
    for i, classifier in enumerate(classifiers):
        importances[i, :] = classifier.feature_importances_
        importances_std[i, :] = np.std([tree.feature_importances_ for tree in classifier.estimators_],
             axis=0)

    mean_importances = np.mean(importances, axis=0)
    std_importances = np.mean(importances_std, axis=0)
    feats = zip(var_names, mean_importances, std_importances)

    # Remove non-important feats:
    feats = [feat for feat in feats if feat[1] > 0.0]

    feats.sort(reverse=True, key=lambda x: x[1])
    print tabulate(feats, headers=['Variable', 'Mean', 'Std'])
    feats.sort(reverse=False, key=lambda x: x[1])

    # Plot the feature importances of the classifier
    plt.figure()
    plt.title("Gini Importance")
    y_pos = np.arange(len(feats))
    plt.barh(y_pos, width=zip(*feats)[1], height=0.5, color='r', xerr=zip(*feats)[2], align="center")
    plt.yticks(y_pos, zip(*feats)[0])
    plt.show()


def main(job_id, params):
    """
    Main hook for Spearmint.
    :param job_id:
    :param params:
    :return:
    """
    score = experiment(params=params, classifier_fn=adaboost, n=default_n_trials, test=False, **dataset_args[default_dataset])
    return score


if __name__ == "__main__":
    # Entry point when running the script manually. Not run by Spearmint.
    job_id = 0
    params = {
        'n_estimators': 262,
        'log_learning_rate': -5.0,
        'max_depth': 1
    }
    evaluate(params=params, classifier_fn=adaboost, n=default_n_trials, test=False, model_metrics=model_metrics, **dataset_args[default_dataset])

