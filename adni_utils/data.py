import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import tables as tb


def balanced_indices(y, sample=True):
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
    if sample:
        balanced_inds = np.concatenate([np.random.choice(ind, replace=False, size=min_class_size) for ind in inds])
    else:
        balanced_inds = np.concatenate([ind[0:min_class_size] for ind in inds])
    np.random.shuffle(balanced_inds)  # in-place shuffle
    return balanced_inds


def balance_set(X, y, sample=True):
    """
    Balance a dataset by using only as many samples per class as the minimum number of samples per class.
    :param X:
    :param y:
    :return:
    """
    inds = balanced_indices(y, sample)
    return X[inds, :], y[inds]


def load_segmentations(**kwargs):
    """

    :param train_data:
    :param test_data:
    :param valid_data:
    :param dataset:
    :param side:
    :param structure:
    :param use_fused:
    :return:
    """
    train_data = kwargs['train_data']
    test_data = kwargs['test_data']
    valid_data = kwargs['valid_data']
    dataset = kwargs['dataset']
    omit_class = kwargs['omit_class']
    side = kwargs['side']
    structure = kwargs['structure']
    use_fused = kwargs['use_fused']

    fused = '_fused' if use_fused else ''

    if 'b' in side:
        X_l = train_data.get_node('/l_{}_features{}'.format(structure.lower(), fused))[:]
        X_r = train_data.get_node('/r_{}_features{}'.format(structure.lower(), fused))[:]
        X_vl = valid_data.get_node('/l_{}_features{}'.format(structure.lower(), fused))[:]
        X_vr = valid_data.get_node('/r_{}_features{}'.format(structure.lower(), fused))[:]
        X_tl = test_data.get_node('/l_{}_features{}'.format(structure.lower(), fused))[:]
        X_tr = test_data.get_node('/r_{}_features{}'.format(structure.lower(), fused))[:]

        X = np.concatenate([X_l, X_r], axis=1)
        X_v = np.concatenate([X_vl, X_vr], axis=1)
        X_t = np.concatenate([X_tl, X_tr], axis=1)
    else:
        X = train_data.get_node('/{}_{}_features{}'.format(side, structure.lower(), fused))[:]
        X_v = valid_data.get_node('/{}_{}_features{}'.format(side, structure.lower(), fused))[:]
        X_t = test_data.get_node('/{}_{}_features{}'.format(side, structure.lower(), fused))[:]

    label_node = 'labels' if omit_class == None else 'label'

    y = train_data.get_node('/{}_fused'.format(label_node))[:]
    y_v = valid_data.get_node('/{}_fused'.format(label_node))[:]
    y_t = test_data.get_node('/{}_fused'.format(label_node))[:]

    default_var_names = [str(x) for x in range(X.shape[1])]

    return X, X_v, X_t, y, y_v, y_t, default_var_names


def load_cortical(**kwargs):
    """

    :param kwargs:
    :return:
    """
    train_data = kwargs['train_data']
    test_data = kwargs['test_data']
    valid_data = kwargs['valid_data']
    omit_class = kwargs.get('omit_class')

    ct_data = pd.read_csv('/projects/nikhil/ADNI_prediction/input_datasets/CT/scans_AAL.csv')
    cortical_variables = list(ct_data.columns[1:])


    X = train_data.get_node('/features')[:]
    X_v = valid_data.get_node('/features')[:]
    X_t = test_data.get_node('/features')[:]
    y = train_data.get_node('/labels')[:]
    y_v = valid_data.get_node('/labels')[:]
    y_t = test_data.get_node('/labels')[:]

    if omit_class != None:
        inds = np.where(y != omit_class)
        inds_v = np.where(y_v != omit_class)
        inds_t = np.where(y_t != omit_class)
        X = X[inds]
        X_v = X_v[inds_v]
        X_t = X_t[inds_t]

        if omit_class == 0:
            trans_fn = lambda x: x - 1
        elif omit_class == 1:
            trans_fn = lambda x: x / 2
        else:
            trans_fn = lambda x: x

        y = trans_fn(y[inds])
        y_v = trans_fn(y_v[inds_v])
        y_t = trans_fn(y_t[inds_t])

    return X, X_v, X_t, y, y_v, y_t, cortical_variables


def load_matrices(**kwargs):
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
    def map_hc_class_to_file(omit_class):
        if omit_class == None: return 'ad_mci_cn'
        if omit_class == 0: return 'mci_cn'
        if omit_class == 1: return 'ad_mci'
        if omit_class == 2: return 'ad_cn'

    data_fns = {
        'mci_cn': load_segmentations,
        'ad_cn': load_segmentations,
        'ad_mci_cn': load_segmentations,
        'ADNI_Cortical_Features': load_cortical,
    }

    dataset = kwargs.get('dataset')
    fold = kwargs.get('fold')
    source_path = kwargs.get('source_path', )
    normalize_data = kwargs.get('normalize_data', True)
    balance = kwargs.get('balance', True)
    omit_class = kwargs.get('omit_class', None)

    if 'hc' in dataset:
        filename = map_hc_class_to_file(omit_class)
    else:
        filename = dataset

    # balanced = '_balanced' if balance else ''

    train_data_file = '{}_{}{}.h5'.format(filename, 'train', fold)
    valid_data_file = '{}_{}{}.h5'.format(filename, 'valid', fold)
    test_data_file = '{}_{}{}.h5'.format(filename, 'test', fold)

    d = tb.open_file(source_path + train_data_file)
    d_valid = tb.open_file(source_path + valid_data_file)
    d_test = tb.open_file(source_path + test_data_file)

    matrix_fn = data_fns[dataset]

    X, X_v, X_t, y, y_v, y_t, var_names = matrix_fn(train_data=d, test_data=d_test, valid_data=d_valid, **kwargs)

    if balance:
        X, y = balance_set(X, y)
        X_v, y_v = balance_set(X_v, y_v)
        X_t, y_t = balance_set(X_t, y_t, sample=False)  # Deterministic balanced split for testing/comparisons?

    if normalize_data:
        X = normalize(X)
        X_v = normalize(X_v)
        X_t = normalize(X_t)

    d.close()
    d_valid.close()
    d_test.close()

    return X, X_v, X_t, y, y_v, y_t, var_names
