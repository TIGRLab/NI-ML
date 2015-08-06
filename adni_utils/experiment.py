import logging
import numpy as np
import sklearn
from matplotlib import pyplot as plt
from adni_utils.data import load_segmentation_dataset_matrices
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    target_names = [ 'ad', 'cn', 'mci']
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def three_class_piecewise(y, y_hat):
    if abs(y - y_hat) == 0: return 0.0
    if abs(y - y_hat) == 1: return 1.0
    if abs(y - y_hat) == 2: return 0.5


def binary_accuracy(y, y_hat):
    return sklearn.metrics.accuracy_score(y, y_hat)


def three_way_accuracy(y, y_hat):
    loss = [three_class_piecewise(i, j) for i, j in zip(y, y_hat)]
    return np.mean(loss)


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


    y_hat = classifier.predict(X_v)
    score_fn = binary_accuracy if n_classes == 2 else three_way_accuracy
    vscore = score_fn(y_v, y_hat)

    return vscore, y_hat


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
        X, X_v, X_t, y, y_v, y_t = load_segmentation_dataset_matrices(source_path, fold, side, dataset, structure, use_fused=use_fused,
                                                 normalize_data=True, balance=balance)

        logging.info('Training Sample Size: {}'.format(X.shape[0]))
        logging.info('Validation Sample Size: {}'.format(X_v.shape[0]))

        vscore, v_preds = experiment_on_fold(params, X, X_v, y, y_v, classifier_fn)
        logging.info('Validation Score: {}'.format(vscore))
        total_vscore += vscore

    # total_tscore /= len(folds)
    total_vscore /= len(folds)

    logging.info('Avg Validation Score: {}'.format(total_vscore))
    logging.info('Avg Spearmint Error: {}'.format(1 - total_vscore))

    # Minimize error (for spearmint):
    return (1.0 - total_vscore)


def test(params, classifier_fn, structure, side, dataset, source_path, use_fused, balance, n=10):
    """
    Test the classifier's predictive ability on the held-out test data set by averaging results from n trials.
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

    X, X_v, X_t, y, y_v, y_t = load_segmentation_dataset_matrices(source_path,
                                             fold='',
                                             side=side,
                                             dataset=dataset,
                                             structure=structure,
                                             use_fused=use_fused,
                                             normalize_data=True,
                                             balance=balance)
    logging.info('Training Sample Size: {}'.format(X.shape[0]))
    logging.info('Test Sample Size: {}'.format(X_t.shape[0]))

    score = []
    preds = []
    labels = []
    for i in range(n):
        tscore, tpreds = experiment_on_fold(params, X, X_t, y, y_t, classifier_fn, test=True)
        score.append(tscore)
        preds.append(tpreds)
        labels.append(y_t)

    mean_score = np.mean(score)
    var_score = np.var(score)
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    print preds

    print 'Held out Test Set Mean Error on {} trials: {}'.format(n, mean_score)
    print 'Held out Test Set Variance on {} trials: {}'.format(n, var_score)


    cm = sklearn.metrics.confusion_matrix(labels, preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print cm_normalized
    plot_confusion_matrix(cm_normalized)
    plt.show()

    return tscore