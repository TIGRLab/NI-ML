import logging
import numpy as np
import sklearn
from matplotlib import pyplot as plt
from adni_utils.data import load_matrices


def binary_accuracy(y, y_hat):
    return sklearn.metrics.accuracy_score(y, y_hat)


def metrics(classifier, X, y):
    """
    Return some metrics for the trained classifier on some data.
    :param classifier:
    :param X:
    :param y:
    :return:
    """
    y_hat = classifier.predict(X)
    acc = sklearn.metrics.accuracy_score(y, y_hat)
    prec = sklearn.metrics.precision_score(y, y_hat)
    rec = sklearn.metrics.recall_score(y, y_hat)
    f1 = sklearn.metrics.f1_score(y, y_hat)
    return acc, prec, rec, f1



def spearmint_score_fn(classifier, train_acc, val_acc, n_classes):
    """
    Loss function used for Spearmint optimization.

    Lower is better. :)
    :param train_acc:
    :param val_acc:
    :return:
    """
    fit_score = abs(train_acc - val_acc)
    val_error = 1 - val_acc
    spearmint_score = fit_score + val_error
    return spearmint_score


def three_way_accuracy(y, y_hat):
    """
    Return a weighted 3-class accuracy score.
    We multiply by (6 / 5) * mean(loss) to normalize between 0 < loss <  1.0, since
    the upper bound on the loss when scoring a class balanced set is (1 * N/3) + (1 * N/3) + (0.5 * N/3) = (5/6)*N
    :param y:
    :param y_hat:
    :return:
    """

    def three_class_piecewise(y, y_hat):
        """
        Weigh AD <-> CN classification mistakes with double the weight of MCI <-> CN or AD <-> CN
        :param y:
        :param y_hat:
        :return:
        """
        if abs(y - y_hat) == 0: return 0.0
        if abs(y - y_hat) == 1: return 1.0
        if abs(y - y_hat) == 2: return 0.5

    loss = [three_class_piecewise(i, j) for i, j in zip(y, y_hat)]
    return 1 - (6 / 5.0) * np.mean(loss)


def experiment_on_fold(X, X_held_out, y, y_held_out, **kwargs):
    """
    Fit a classifier on the given training dataset and score it on a validation set.
    :param params:
    :param X:
    :param X_held_out:
    :param y:
    :param y_held_out:
    :param classifier_fn:
    :param test:
    :return:
    """
    params = kwargs.get('params')
    classifier_fn = kwargs.get('classifier_fn')

    n_classes = np.unique(y).shape[0]
    classifier, model = classifier_fn(params, n_classes)

    classifier.fit(X, y)

    training_accuracy = classifier.score(X, y)
    held_out_predictions = classifier.predict(X_held_out)
    held_out_accuracy = classifier.score(X_held_out, y_held_out)
    spearmint_score = spearmint_score_fn(classifier, training_accuracy, held_out_accuracy, n_classes)

    return spearmint_score, held_out_accuracy, held_out_predictions, training_accuracy


def unpack_experimental_params(**kwargs):
    """
    Unpack keyword arguments for experiment method.
    :param kwargs:
    :return:
    """
    source_path = kwargs.get('source_path')
    params = kwargs.get('params')
    classifier_fn = kwargs.get('classifier_fn')
    dataset = kwargs.get('dataset')
    load_fn = kwargs.get('load_fn', load_matrices)
    structure = kwargs.get('structure')
    side = kwargs.get('side')
    balance = kwargs.get('balance', True)
    use_fused = kwargs.get('use_fused', True)
    n = kwargs.get('n', 1)
    normalize_data = kwargs.get('normalize_data', True)
    test = kwargs.get('test', False)
    folds = kwargs.get('folds', [''])

    return source_path, params, classifier_fn, dataset, load_fn, structure, side, folds, use_fused, balance, normalize_data, n, test


def n_trials_fn(X, X2, X3, y, y2, y3, n, test=False):
    X_train = X
    X_holdout = X3 if test else X2
    y_train = y
    y_holdout = y3 if test else y2
    return [(X_train, X_holdout, y_train, y_holdout) for i in range(n)]


def leave_one_out_fn(X, X2, X3, y, y2, y3, n, test=False):
    X = np.concatenate([X, X2],axis=0)
    y = np.concatenate([y, y2],axis=0)
    N = X.shape[0]
    splits = sklearn.cross_validation.KFold(N, n_folds=N, random_state=0)
    return [(X[tinds], X[vinds], y[tinds], y[vinds]) for tinds, vinds in splits]


cross_val_fn_map = {
    'n_trials': n_trials_fn,
    'leave_one_out': leave_one_out_fn,
}


def experiment(**kwargs):
    """
    Train and test the classifier's predictive ability on the held-out validation data set by averaging results from n trials.
    :param params:
    :param classifier_fn:
    :param structure:
    :param side:
    :param dataset:
    :param folds:
    :param source_path:
    :param use_fused:
    :param balance:
    :return: X, X_valid, X_test, y, y_valid, y_test
    """
    logfile = kwargs.get('logfile', './output/spearmint.log')
    logging.basicConfig(level=logging.INFO, format="%(message)s", filemode='a', filename=logfile)

    load_fn = kwargs.get('load_fn', load_matrices)
    folds = kwargs.get('folds', [''])
    params = kwargs.get('params')
    n = kwargs.get('n', 1)
    job_id = kwargs.get('job_id',0)
    cross_val = kwargs.get('cross_val_fn', 'n_trials')
    cross_val_fn = cross_val_fn_map[cross_val]

    score = []
    train = []
    acc = []
    preds = []
    labels = []

    for j, fold in enumerate(folds):
        X, X2, X3, y, y2, y3, var_names = load_fn(fold=fold, **kwargs)
        for X, X_held_out, y, y_held_out in cross_val_fn(X, X2, X3, y, y2, y3, n):
            held_out_spearmint_score, held_out_accuracy, held_out_predictions, training_accuracy = experiment_on_fold(
                X=X,
                X_held_out=X_held_out,
                y=y,
                y_held_out=y_held_out,
                **kwargs)

            score.append(held_out_spearmint_score)
            acc.append(held_out_accuracy)
            preds.append(held_out_predictions)
            labels.append(y_held_out)
            train.append(training_accuracy)

    mean_spearmint_score = np.mean(score)
    std_score = np.std(score)
    mean_val = np.mean(acc)
    std_val = np.std(acc)
    mean_train = np.mean(training_accuracy)
    std_train = np.std(training_accuracy)
    mean_spearmint_score += std_val

    # alphabetize params:
    alpha_params = params.items()
    alpha_params.sort()

    param_log = map(lambda x: x[0] if isinstance(x, np.ndarray) else x, zip(*alpha_params)[1])
    logging.info('{:08d}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\t'.format(job_id, mean_spearmint_score, std_score, mean_val, std_val, mean_train,
                                             std_train) + ''.join('{:.8f}\t'.format(p) for p in param_log))

    # Final spearmint score has std term: more stable algos are better
    return mean_spearmint_score
