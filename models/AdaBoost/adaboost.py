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


def adaboost(params, n_classes):
    learning_rate = np.exp(params['log_learning_rate'])
    n_estimators = params['n_estimators']
    # Min number of decision tree branches required to classify n_classes
    min_max_depth = int(np.ceil(np.log(n_classes)))

    stump = DecisionTreeClassifier(max_depth=min_max_depth)

    classifier = AdaBoostClassifier(stump,
                             algorithm="SAMME.R",
                             n_estimators=n_estimators,
                             learning_rate=learning_rate)
    return classifier, 'AdaBoost'


def main(job_id, params, side=default_side, dataset=default_dataset):
    """
    Main hook for Spearmint.
    :param job_id:
    :param params:
    :return:
    """
    logging.basicConfig(level=logging.INFO)
    score = experiment(params=params, classifier_fn=adaboost, structure=structure, side=side, dataset=dataset,
                       folds=folds, source_path=source_path, use_fused=use_fused, balance=balance, n=n_trials, test=False)
    return score

if __name__ == "__main__":
    # Entry point when running the script manually. Not run by Spearmint.
    held_out_test = True
    job_id = 0
    params = {
        'log_learning_rate': -4.87273849,
        'n_estimators': 500
    }
    if held_out_test:
        experiment(params=params, classifier_fn=adaboost, structure=structure, side=default_side, dataset=default_dataset,
                   folds=folds, source_path=source_path, use_fused=use_fused, balance=balance, n=1, test=held_out_test)
    else:
        for side in sides:
            for dataset in adni_datasets:
                main(job_id, params, side, dataset)