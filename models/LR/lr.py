from sklearn import linear_model, decomposition
import sys
import logging
from sklearn.pipeline import Pipeline

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


def pca_lr(params, n_classes):
    C = params['C']
    n_components = params['n_components']
    mclass = 'multinomial' if n_classes > 2 else 'ovr'
    solver = 'lbfgs' if n_classes > 2 else 'liblinear'

    logistic = linear_model.LogisticRegression(C=C, verbose=1, multi_class=mclass, solver=solver)

    pca = decomposition.RandomizedPCA(n_components=n_components)
    pca_lr_classifier = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
    return pca_lr_classifier, 'PCA Logistic Regression'


def main(job_id, params, side=default_side, dataset=default_dataset):
    """
    Main hook for Spearmint.
    :param job_id:
    :param params:
    :return:
    """
    logging.basicConfig(level=logging.INFO)
    score = experiment(params=params, classifier_fn=pca_lr, structure=structure, side=side, dataset=dataset,
                       folds=folds, source_path=source_path, use_fused=use_fused)
    return score


if __name__ == "__main__":
    # Entry point when running the script manually. Not run by Spearmint.
    job_id = 0
    arguments = {
        'n_components': 16,
        'C': 0.5,
    }
    for side in sides:
        for dataset in adni_datasets:
            main(job_id, arguments, side, dataset)
