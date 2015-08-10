import logging
import numpy as np
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


# def load_segmentations(train_data, test_data, valid_data, dataset, side, structure, use_fused):
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

    label_node = 'labels' if 'ad_mci_cn' in dataset else 'label'

    y = train_data.get_node('/{}_fused'.format(label_node))[:]
    y_v = valid_data.get_node('/{}_fused'.format(label_node))[:]
    y_t = test_data.get_node('/{}_fused'.format(label_node))[:]
    return X, X_t, X_v, y, y_t, y_v


def load_cortical(**kwargs):
    """

    :param kwargs:
    :return:
    """
    train_data = kwargs['train_data']
    test_data = kwargs['test_data']
    valid_data = kwargs['valid_data']
    side = kwargs['side']
    structure = kwargs['structure']

    X = train_data.get_node('/features'.format(side, structure.lower()))[:]
    X_v = valid_data.get_node('/features'.format(side, structure.lower()))[:]
    X_t = test_data.get_node('/features'.format(side, structure.lower()))[:]
    y = train_data.get_node('/labels')[:]
    y_v = valid_data.get_node('/labels')[:]
    y_t = test_data.get_node('/labels')[:]
    return X, X_t, X_v, y, y_t, y_v


def load_matrices(source_path, fold, side, dataset, structure, use_fused=False,
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
    data_fns = {
        'ad_mci_cn': load_segmentations,
        'ADNI_cortical_Features': load_cortical,
    }

    # balanced = '_balanced' if balance else ''

    train_data_file = '{}_{}{}.h5'.format(dataset, 'train', fold)
    valid_data_file = '{}_{}{}.h5'.format(dataset, 'valid', fold)
    test_data_file = '{}_{}{}.h5'.format(dataset, 'test', fold)

    d = tb.open_file(source_path + train_data_file)
    d_valid = tb.open_file(source_path + valid_data_file)
    d_test = tb.open_file(source_path + test_data_file)

    matrix_fn = data_fns[dataset]

    # Specific to segmentation:
    X, X_t, X_v, y, y_t, y_v = matrix_fn(train_data=d, test_data=d_test, valid_data=d_valid, dataset=dataset, side=side,
                                         structure=structure, use_fused=use_fused)

    # Non-specific:
    if balance:
        logging.info('Balancing Classes...')
        X, y = balance_set(X, y)
        X_v, y_v = balance_set(X_v, y_v)
        X_t, y_t = balance_set(X_t, y_t, sample=False) # Deterministic balanced split for testing/comparisons?

    if normalize_data:
        logging.info('Normalizing Matrices...')
        X = normalize(X)
        X_v = normalize(X_v)
        X_t = normalize(X_t)

    d.close()
    d_valid.close()
    d_test.close()

    return X, X_v, X_t, y, y_v, y_t
