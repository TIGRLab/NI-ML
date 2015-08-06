import logging
import numpy as np
from sklearn.preprocessing import normalize
import tables as tb


def balanced_indices(y):
    """
    Returns a shuffled set of indices for a class-balanced set of samples.
    :param y:
    :return:
    """
    n_classes = np.unique(y).shape[0]
    inds = []
    lengths = []
    for c in range(n_classes):
        ind = np.where(y == c)[0]
        inds.append(ind)
        lengths.append(len(ind))
    min_class_size = np.min(lengths)
    balanced_inds = np.concatenate([np.random.choice(ind, replace=False,size=min_class_size) for ind in inds])
    np.random.shuffle(balanced_inds)  # in-place shuffle
    return balanced_inds


def balance_set(X, y):
    """
    Balance a dataset by using only as many samples per class as the minimum number of samples per class.
    :param X:
    :param y:
    :return:
    """
    inds = balanced_indices(y)
    return X[inds, :], y[inds]


def load_segmentation_dataset_matrices(source_path, fold, side, dataset, structure, use_fused=False,
                                       normalize_data=True, balance=True):
    """
    Load and return all data matrices for the given data set.
    :param source_path:
    :param fold:
    :param side:
    :param dataset:
    :param structure:
    :param use_fused:
    :param normalize_data:
    :param balance:
    :return:
    """
    logging.info('Loading {} data...'.format(dataset))

    #balanced = '_balanced' if balance else ''

    train_data_file = '{}_{}{}.h5'.format(dataset, 'train', fold)
    valid_data_file = '{}_{}{}.h5'.format(dataset, 'valid', fold)
    test_data_file = '{}_{}{}.h5'.format(dataset, 'test', fold)

    d = tb.open_file(source_path + train_data_file)
    d_valid = tb.open_file(source_path + valid_data_file)
    d_test = tb.open_file(source_path + test_data_file)

    fused = '_fused' if use_fused else ''

    if 'b' in side:
        X_l = d.get_node('/l_{}_features{}'.format(structure.lower(), fused))[:]
        X_r = d.get_node('/r_{}_features{}'.format(structure.lower(), fused))[:]
        X_vl = d_valid.get_node('/l_{}_features{}'.format(structure.lower(), fused))[:]
        X_vr = d_valid.get_node('/r_{}_features{}'.format(structure.lower(), fused))[:]
        X_tl = d_test.get_node('/l_{}_features{}'.format(structure.lower(), fused))[:]
        X_tr = d_test.get_node('/r_{}_features{}'.format(structure.lower(), fused))[:]

        X = np.concatenate([X_l, X_r], axis=1)
        X_v = np.concatenate([X_vl, X_vr], axis=1)
        X_t = np.concatenate([X_tl, X_tr], axis=1)
    else:
        X = d.get_node('/{}_{}_features{}'.format(side, structure.lower(), fused))[:]
        X_v = d_valid.get_node('/{}_{}_features{}'.format(side, structure.lower(), fused))[:]
        X_t = d_test.get_node('/{}_{}_features{}'.format(side, structure.lower(), fused))[:]

    label_node = 'labels' if 'ad_mci_cn' in dataset else 'label'

    y = d.get_node('/{}_fused'.format(label_node))[:]
    y_v = d_valid.get_node('/{}_fused'.format(label_node))[:]
    y_t = d_test.get_node('/{}_fused'.format(label_node))[:]

    if balance:
        logging.info('Balancing Classes...')
        X, y = balance_set(X, y)
        X_v, y_v = balance_set(X_v, y_v)
        X_t, y_t = balance_set(X_t, y_t)

    if normalize_data:
        logging.info('Normalizing Matrices...')
        X = normalize(X)
        X_v = normalize(X_v)
        X_t = normalize(X_t)

    d.close()
    d_valid.close()
    d_test.close()

    return X, X_v, X_t, y, y_v, y_t
