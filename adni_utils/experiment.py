import logging
import numpy as np
import sklearn
from matplotlib import pyplot as plt
from adni_utils.data import load_matrices


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, target_names=['ad', 'cn', 'mci']):
    """
    Sklearn-style confusion matrix.
    :param cm:
    :param title:
    :param cmap:
    :return:
    """
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
    """
    Weigh AD <-> CN classification mistakes with double the weight of MCI <-> CN or AD <-> CN
    :param y:
    :param y_hat:
    :return:
    """
    if abs(y - y_hat) == 0: return 0.0
    if abs(y - y_hat) == 1: return 1.0
    if abs(y - y_hat) == 2: return 0.5


def binary_accuracy(y, y_hat):
    return sklearn.metrics.accuracy_score(y, y_hat)


def three_way_accuracy(y, y_hat):
    """
    Return a weighted 3-class accuracy score.
    We multiply by (6 / 5) * mean(loss) to normalize between 0 < loss <  1.0, since
    the upper bound on the loss when scoring a class balanced set is (1 * N/3) + (1 * N/3) + (0.5 * N/3) = (5/6)*N
    :param y:
    :param y_hat:
    :return:
    """
    logging.info('Using 3-way weighted scoring metric')
    loss = [three_class_piecewise(i, j) for i, j in zip(y, y_hat)]
    return 1 - (6 / 5.0) * np.mean(loss)


def experiment_on_fold(params, X, X_held_out, y, y_held_out, classifier_fn, test=False):
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
    n_classes = np.unique(y).shape[0]
    classifier, model = classifier_fn(params, n_classes)

    logging.info('Fitting {} on {} classes'.format(model, n_classes))
    classifier.fit(X, y)
    logging.info('{} accuracy on training data.'.format(classifier.score(X, y)))
    trial = 'Validating' if not test else 'Testing'
    logging.info('{} {} on held-out set'.format(trial, model))

    held_out_predictions = classifier.predict(X_held_out)
    held_out_accuracy = classifier.score(X_held_out, y_held_out)
    logging.info('{} accuracy on held-out data.'.format(held_out_accuracy))
    score_fn = binary_accuracy if n_classes == 2 else three_way_accuracy
    held_out_score = score_fn(y_held_out, held_out_predictions)

    return held_out_score, held_out_accuracy, held_out_predictions


def experiment(params, classifier_fn, structure, side, dataset, folds, source_path, use_fused, balance, n=1,
               test=False):
    """
    Test the classifier's predictive ability on the held-out test data set by averaging results from n trials.
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
    logging.info('Testing Model on side {} of test dataset {}'.format(side, dataset))
    logging.info('Using Parameters: ')
    logging.info(params)
    score = []
    acc = []
    preds = []
    labels = []

    for j, fold in enumerate(folds):
        for i in range(n):
            logging.info('Trial {} on fold {}'.format(i, j))
            X, X_v, X_t, y, y_v, y_t = load_matrices(source_path=source_path,
                                                     fold=fold,
                                                     side=side,
                                                     dataset=dataset,
                                                     structure=structure,
                                                     use_fused=use_fused,
                                                     normalize_data=True,
                                                     balance=balance)
            if test:
                X_held_out = X_t
                y_held_out = y_t
            else:
                X_held_out = X_v
                y_held_out = y_v

            held_out = 'Test' if test else 'Validation'

            logging.info('Training Sample Size: {}'.format(X.shape[0]))
            logging.info('{} Sample Size: {}'.format(held_out, X_held_out.shape[0]))

            held_out_score, held_out_accuracy, held_out_predictions = experiment_on_fold(params=params,
                                                                                         X=X,
                                                                                         X_held_out=X_held_out,
                                                                                         y=y,
                                                                                         y_held_out=y_held_out,
                                                                                         classifier_fn=classifier_fn,
                                                                                         test=test)
            score.append(held_out_score)
            acc.append(held_out_accuracy)
            preds.append(held_out_predictions)
            labels.append(y_held_out)

    mean_score = np.mean(score)
    var_score = np.var(score)
    mean_acc = np.mean(acc)
    var_acc = np.var(acc)


    preds = np.concatenate(preds)
    labels = np.concatenate(labels)

    logging.info('Held out Set Mean Classification Accuracy on {} trials: {}'.format(n, mean_acc))
    logging.info('Held out Set Classification Accuracy Variance on {} trials: {}'.format(n, var_acc))
    logging.info('Held out Set Mean Class-Weighted Accuracy on {} trials: {}'.format(n, mean_score))
    logging.info('Held out Set Weighted Class-Weighted Accuracy Variance on {} trials: {}'.format(n, var_score))
    logging.info('Parameters used on this run: ')
    logging.info(params)

    if test:
        cm = sklearn.metrics.confusion_matrix(labels, preds)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print cm_normalized
        plot_confusion_matrix(cm_normalized, target_names=['ad', 'cn', 'mci'])
        plt.show()

    return 1 - mean_score
