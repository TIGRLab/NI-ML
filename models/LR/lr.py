from sklearn import linear_model, decomposition
import sys
import logging
from sklearn.pipeline import Pipeline

root = '/projects/francisco/repositories/NI-ML/'
sys.path.insert(0, root)

# Load repo-specific imports:
from adni_utils.experiment import experiment
from adni_utils.dataset_constants import *


def pca_lr(params, n_classes):
    C = params['C']
    n_components = params['n_components']
    mclass = 'multinomial' if n_classes > 2 else 'ovr'
    solver = 'lbfgs' if n_classes > 2 else 'liblinear'

    logistic = linear_model.LogisticRegression(C=C, verbose=1, multi_class=mclass, solver=solver)

    pca = decomposition.RandomizedPCA(n_components=n_components)
    pca_lr_classifier = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
    return pca_lr_classifier, 'PCA Logistic Regression'


def main(job_id, params, side=evaluation_side, dataset=evaluation_dataset):
    """
    Main hook for Spearmint.
    :param job_id:
    :param params:
    :return:
    """
    score = experiment(params=params, classifier_fn=pca_lr, structure=structure, side=side, dataset=dataset,
                       folds=folds, source_path=source_path, use_fused=use_fused, balance=balance, n=n_trials, test=False)
    return score


if __name__ == "__main__":
    # Entry point when running the script manually. Not run by Spearmint.
    held_out_test = True
    job_id = 0
    params = {
        'n_components': 16,
        'C': 0.5,
    }
    if held_out_test:
        experiment(params=params, classifier_fn=pca_lr, structure=structure, side=evaluation_side, dataset=evaluation_dataset,
                    folds=folds, source_path=source_path, use_fused=use_fused, balance=balance, n=n_trials, test=True)
    else:
        for side in sides:
            for dataset in adni_datasets:
                main(job_id, params, side, dataset)
