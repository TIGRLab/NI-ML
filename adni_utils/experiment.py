import logging
import numpy as np
from adni_utils.data import load_matrices


def experiment_on_fold(params, X, X_v, y, y_v, classifier_fn, test=False):
    """
    Fit a classifier on the given training dataset and score it on a validation set.
    :param params:
    :param X:
    :param X_v:
    :param y:
    :param y_v:
    :param classifier_fn:
    :param test:
    :return:
    """
    n_classes = np.unique(y).shape[0]
    classifier, model = classifier_fn(params, n_classes)

    logging.info('Fitting {} on {} classes'.format(model, n_classes))
    classifier.fit(X, y)

    trial = 'Validating' if not test else 'Testing'
    logging.info('{} {} on held-out set'.format(trial, model))
    vscore = classifier.score(X_v, y_v)

    return vscore


def experiment(params, classifier_fn, structure, side, dataset, folds, source_path, use_fused, balance):
    """
    Run a full experiment on the classifier returned by classifier_fn.
    :param params:
    :param classifier_fn:
    :param structure:
    :param side:
    :param dataset:
    :param folds:
    :param source_path:
    :param use_fused:
    :param balance:
    :return:
    """
    logging.basicConfig(level=logging.INFO)
    logging.info('Running Experiment on side {} of {} structure from {} dataset:'.format(side, structure, dataset))
    logging.info('Using Parameters: ')
    logging.info(params)
    total_vscore = 0.0

    for i, fold in enumerate(folds):
        logging.info("Fold {}:".format(i))
        X, X_v, X_t, y, y_v, y_t = load_matrices(source_path, fold, side, dataset, structure, use_fused=use_fused,
                                                 normalize_data=True, balance=balance)

        logging.info('Training Sample Size: {}'.format(X.shape[0]))
        logging.info('Validation Sample Size: {}'.format(X_v.shape[0]))

        vscore = experiment_on_fold(params, X, X_v, y, y_v, classifier_fn)
        logging.info('Validation Score: {}'.format(vscore))
        total_vscore += vscore

    # total_tscore /= len(folds)
    total_vscore /= len(folds)

    logging.info('Avg Validation Score: {}'.format(total_vscore))
    logging.info('Avg Spearmint Error: {}'.format(1 - total_vscore))

    # Minimize error (for spearmint):
    return (1.0 - total_vscore)


def test(params, classifier_fn, structure, side, dataset, source_path, use_fused, balance):
    """
    Test the classifier's predictive ability on the held-out test data set.
    :param params:
    :param classifier_fn:
    :param structure:
    :param side:
    :param dataset:
    :param source_path:
    :param use_fused:
    :param balance:
    :return:
    """
    logging.basicConfig(level=logging.INFO)
    logging.info('Testing Model on side {} of test dataset {}'.format(side, dataset))
    logging.info('Using Parameters: ')
    logging.info(params)
    X, X_v, X_t, y, y_v, y_t = load_matrices(source_path,
                                             fold='',
                                             side=side,
                                             dataset=dataset,
                                             structure=structure,
                                             use_fused=use_fused,
                                             normalize_data=True,
                                             balance=balance)
    logging.info('Training Sample Size: {}'.format(X.shape[0]))
    logging.info('Test Sample Size: {}'.format(X_t.shape[0]))

    tscore = experiment_on_fold(params, X, X_t, y, y_t, classifier_fn, test=True)
    print 'Held out Test Set Error: {}'.format(tscore)
    return tscore