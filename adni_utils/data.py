import numpy as np
from sklearn.preprocessing import normalize
import tables as tb

def load_matrices(source_path, fold, side, dataset, structure, use_fused=False, normalize_data=True):
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

    if normalize_data:
        X = normalize(X)
        X_v = normalize(X_v)
        X_t = normalize(X_t)

    label_node = 'labels' if 'ad_mci_cn' in dataset else 'label'

    y = d.get_node('/{}_fused'.format(label_node))[:]
    y_v = d_valid.get_node('/{}_fused'.format(label_node))[:]
    y_t = d_test.get_node('/{}_fused'.format(label_node))[:]

    d.close()
    d_valid.close()
    d_test.close()

    return X, X_v, X_t, y, y_v, y_t
