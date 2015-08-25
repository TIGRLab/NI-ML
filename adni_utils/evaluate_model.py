import os
import datetime
import numpy as np
import sklearn
from matplotlib import pyplot as plt
from adni_utils.data import load_matrices
from adni_utils.experiment import unpack_experimental_params, metrics, cross_val_fn_map
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

    dataset = kwargs.get('dataset')
    folds = kwargs.get('folds', [''])
    n = kwargs.get('n')
    load_fn = kwargs.get('load_fn', load_matrices)
    classifier_fn = kwargs.get('classifier_fn')
    params = kwargs.get('params')
    class_names = kwargs.get('class_names')
    model_metrics_fn = kwargs.get('model_metrics')

    if class_names == None:
        class_names = class_name_map[dataset]

    print 'Running model on {} data test set for {} trials'.format(dataset, n)

    omit_class = kwargs.get('omit_class')

    if omit_class != None: # I know: awkward.
        print 'Omitting class {} ({})'.format(omit_class, class_names[omit_class])
        del class_names[omit_class]
        print 'Classifying between {}'.format(class_names)


    cross_val = kwargs.get('cross_val_fn', 'n_trials')
    cross_val_fn = cross_val_fn_map[cross_val]
    print 'using {} cross validation function'.format(cross_val)

    train = []
    preds = []
    labels = []
    classifiers = []
    scores = []
    accs = []
    print 'TraAcc ValAcc ValPrec ValRec Valf1'
    for j, fold in enumerate(folds):
        X, X2, X3, y, y2, y3, var_names = load_fn(fold=fold, **kwargs)
        for X, X_held_out, y, y_held_out in cross_val_fn(X, X2, X3, y, y2, y3, n):
                n_classes = np.unique(y).shape[0]
                classifier, model = classifier_fn(params, n_classes)

                classifier.fit(X, y)
                training_accuracy = classifier.score(X, y)
                y_hat = classifier.predict(X_held_out)
                y_score = classifier.predict_proba(X_held_out)
                acc, prec, rec, f1 = metrics(classifier, X_held_out, y_held_out)
                print '{} {} {} {} {}'.format(training_accuracy, acc, prec, rec, f1)
                accs.append(acc)
                preds.append(y_hat)
                labels.append(y_held_out)
                scores.append(y_score)
                train.append(training_accuracy)
                classifiers.append(classifier)

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    scores = np.concatenate(scores)

    print
    print 'Mean Training Acc: {}'.format(np.mean(train))
    print 'Std Training Acc: {}'.format(np.std(train))
    print 'Mean Test Acc: {}'.format(np.mean(accs))
    print 'Std Test Acc: {}'.format(np.std(accs))
    print sklearn.metrics.classification_report(labels, preds,
                                        target_names=class_names)

    cm = sklearn.metrics.confusion_matrix(labels, preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print 'Params:'
    print params

    print 'Number of independent vars: {}'.format(len(var_names))

    plot_path = os.getcwd() + '/plots/'
    if not os.path.isdir(plot_path):
        os.mkdir(plot_path)

    if model_metrics_fn:
        model_metrics_fn(classifiers, var_names)

    print 'CM:'
    print cm_normalized
    plot_confusion_matrix(cm_normalized, target_names=class_names)
    stamp = datetime.datetime.now().strftime("%Y-%M-%d")
    #plt.savefig(plot_path + '{}_{}_{}'.format(model, dataset, stamp))
    plt.show()
    plt.close('all')

    # ROC
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, scores[:,1], pos_label=1)
    # Save fpr tpr
    #np.savetxt("fpr.csv", fpr, delimiter=",")
    #np.savetxt("tpr.csv", tpr, delimiter=",")
    auc = sklearn.metrics.auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()

    return
