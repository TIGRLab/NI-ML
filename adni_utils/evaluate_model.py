import numpy as np
import sklearn
from matplotlib import pyplot as plt
from adni_utils.experiment import unpack_experimental_params, metrics
from adni_utils.dataset_constants import class_name_map


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


def evaluate(**kwargs):
    """
    Test the classifier's predictive ability on the held-out test data set by averaging results from n trials.

    Collect and display metrics.
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
    source_path, params, classifier_fn, dataset, load_fn, structure, side, folds, use_fused, balance, normalize_data, n, test = unpack_experimental_params(
        **kwargs)

    print 'Running model on {} data test set for {} trials'.format(dataset, n)


    class_names = kwargs.get('class_names')
    if not class_names:
        class_names = class_name_map[dataset]
    omit_class = kwargs.get('omit_class')
    if not omit_class == None: # I know: awkward.
        print 'Omitting class {} ({})'.format(omit_class, class_names[omit_class])
        del class_names[omit_class]
    model_metrics_fn = kwargs.get('model_metrics')
    train = []
    preds = []
    labels = []
    classifiers = []

    accs = []
    for j, fold in enumerate(folds):
        for i in range(n):
            X, X_v, X_t, y, y_v, y_t, var_names = load_fn(source_path=source_path,
                                               fold=fold,
                                               side=side,
                                               dataset=dataset,
                                               structure=structure,
                                               use_fused=use_fused,
                                               normalize_data=True,
                                               balance=balance)

            n_classes = np.unique(y).shape[0]
            classifier, model = classifier_fn(params, n_classes)

            classifier.fit(X, y)
            training_accuracy = classifier.score(X, y)
            y_hat = classifier.predict(X_t)
            acc, prec, rec, f1 = metrics(classifier, X_t, y_t)
            print '{} {} {} {}'.format(acc, prec, rec, f1)
            accs.append(acc)
            preds.append(y_hat)
            labels.append(y_t)
            train.append(training_accuracy)
            classifiers.append(classifier)


    preds = np.concatenate(preds)
    labels = np.concatenate(labels)

    print
    print 'Mean Acc: {}'.format(np.mean(accs))
    print 'Std Acc: {}'.format(np.std(accs))
    print sklearn.metrics.classification_report(labels, preds,
                                        target_names=class_names)

    cm = sklearn.metrics.confusion_matrix(labels, preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print 'Params:'
    print params

    if model_metrics_fn:
        model_metrics_fn(classifiers, var_names)

    print 'CM:'
    print cm_normalized
    plot_confusion_matrix(cm_normalized, target_names=class_names)
    plt.show()

    return
