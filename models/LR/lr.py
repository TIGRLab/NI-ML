from sklearn import linear_model, decomposition
import sys
import logging
from sklearn.pipeline import Pipeline
root = '/projects/francisco/repositories/NI-ML/'
sys.path.insert(0, root)

# Load repo-specific imports:
from adni_utils.experiment import experiment
from adni_utils.dataset_constants import *
from adni_utils.evaluate_model import evaluate


def pca_lr(params, n_classes):
    C = params['C']
    n_components = params['n_components']
    mclass = 'multinomial' if n_classes > 2 else 'ovr'
    solver = 'lbfgs' if n_classes > 2 else 'liblinear'

    logistic = linear_model.LogisticRegression(C=C, verbose=1, multi_class=mclass, solver=solver)

    pca = decomposition.RandomizedPCA(n_components=n_components)
    pca_lr_classifier = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
    return pca_lr_classifier, 'PCA Logistic Regression'


def main(job_id, params):
    """
    Main hook for Spearmint.
    :param job_id:
    :param params:
    :return:
    """
    score = experiment(params=params, classifier_fn=pca_lr, n=default_n_trials, test=False, **dataset_args[default_dataset])

    return score


if __name__ == "__main__":
    # Entry point when running the script manually. Not run by Spearmint.
    held_out_test = True
    job_id = 0
    params = {
        'n_components': 16,
        'C': 0.5,
    }
    evaluate(params=params, classifier_fn=pca_lr, n=default_n_trials, test=False, model_metrics=None, **dataset_args[default_dataset])

