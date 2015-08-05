import logging
import numpy as np
from adni_utils.data import load_matrices


def experiment_on_fold(params, X, X_v, y, y_v, classifier_fn):
    n_classes = np.unique(y).shape[0]
    classifier, model = classifier_fn(params, n_classes)

    logging.info('Fitting {}'.format(model))
    classifier.fit(X, y)

    logging.info('Validating {}'.format(model))
    vscore = classifier.score(X_v, y_v)

    return vscore


def experiment(params, classifier_fn, structure, side, dataset, folds, source_path, use_fused):
    """
    Main hook for Spearmint.
    :param job_id:
    :param params:
    :return:
    """
    logging.basicConfig(level=logging.INFO)
    print 'Running Experiment on side {} of dataset {}:'.format(side, dataset)
    print 'Using Parameters: '
    print params
    total_vscore = 0.0

    for i, fold in enumerate(folds):
        print "Fold {}:".format(i)
        X, X_v, X_t, y, y_v, y_t = load_matrices(source_path, fold, side, dataset, structure, use_fused=use_fused,
                                                 normalize_data=True)
        vscore = experiment_on_fold(params, X, X_v, y, y_v, classifier_fn)
        print 'Validation Score: {}'.format(vscore)
        print
        total_vscore += vscore


    # total_tscore /= len(folds)
    total_vscore /= len(folds)

    print 'Avg Validation Score: {}'.format(total_vscore)
    print


    # Minimize error (for spearmint):
    return (1.0 - total_vscore)