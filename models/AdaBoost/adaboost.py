import sys
import logging
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

root = '/projects/francisco/repositories/NI-ML/'
sys.path.insert(0, root)

# Load repo-specific imports:
from adni_utils.experiment import experiment

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

# List of fold filename extensions to iterate over:
# ie. folds = ['_T1', '_T2', '_T3']
folds = [''] # No folds.

# Change these when running Spearmint experiments which use only the main and don't iterate over datasets or sides:
default_side = 'l' # Valid values: l, r, b
default_dataset = 'ad_mci_cn' # Valid values: mci_cn, ad_cn


def adaboost(params, n_classes):
    learning_rate = params['learning_rate']
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
                       folds=folds, source_path=source_path, use_fused=use_fused)
    return score

if __name__ == "__main__":
    # Entry point when running the script manually. Not run by Spearmint.
    job_id = 0
    params = {
        'learning_rate': 0.05,
        'n_estimators': 250
    }
    for side in sides:
        for dataset in adni_datasets:
            main(job_id, params, side, dataset)
