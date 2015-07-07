from fuel.datasets import H5PYDataset
import h5py
import numpy as np
import tables as tb

datafile = '/projects/jp/adni-autoencoder/combined.h5'
features_template = '/{}_{}_data'
labels_template = '/{}_{}_classes'

sides = [
    'r',
    'l',
]

structures = {
    'EC',
    'HC',
}

input_dims = {
    'HC': {'l': 11427, 'r': 10519, 'b': 10519 + 11427},
    'EC': {'l': 15069, 'r': 14907, 'b': 15069 + 14907},
}


def balanced_mci_indexes(y):
    """
    Reduces the number of mci samples to be half the current set size.
    :param y: Class labels vector for both mci and other class.
    :return: Indexes of mci and other class samples to make a balanced set.
    """
    mcinds = np.where(y == 2)[0]
    oinds = np.where(y != 2)[0]
    reduced_mcinds = mcinds[0: mcinds.shape[0] / 2]
    inds = np.concatenate((reduced_mcinds, oinds))
    np.random.shuffle(inds)
    return inds


# Splits used for the AD-MCI-CN 3-way classification of ADNI data.
# Each item in the 'splits' dict represents one of the 2-way classifiers used on the ADNI dataset:
# ie: 'ad_cn' is the data split for the alzheimer's vs control classifier
# 'ni': The excluded class for the current split: (ie. ad-cn classifier excludes class 2: mci)
# 'labelfn': A function to transform the split's labels: (ie. ad-mci labels transformed from 0-2 to 0-1.
# 'X': The features matrix for the split.
# 'y': The labels vector for the split.
splits = {
    'ad_cn': {
        'ni': 2,
        'labelfn': lambda x: x
    },
    'mci_cn': {
        'ni': 0,
        'labelfn': lambda x: x - 1
    },
    'ad_mci': {
        'ni': 1,
        'labelfn': lambda x: x / 2
    },
}

class_labels = {
    'ad': 0,
    'cn': 1,
    'mci': 2,
}

def load_data(source=datafile):
    """
    Load the existing H5 dataset.

    :param source: Path to the adni H5 dataset.
    """
    FILTERS = tb.Filters(complevel=5, complib='zlib')
    data = tb.open_file(source, mode='r', filters=FILTERS)
    return data


def split_3_way(X, y):
    """
    Given a dataset X and its 3-class labels y, splits it into 3 2-way classification splits.
    :param X: The data features.
    :param y: A 3-class (0,1,2) set of labels.
    :return: The splits dictionary defined above, including the X and y matrices for each split.
    """
    for name, split in splits.items():
        indexes = np.where(y != split['ni'])[0]
        split['X'] = X[indexes]
        split['y'] = split['labelfn'](y[indexes])
    return splits


def make_caffe_file(outfile, X, y, feature_name='features'):
    """
    Create a Caffe-format HDf5 data and labels file.
    :param outfile: A path and filename to write the dataset out to.
    :param set_name: Name of the dataset (ie. 'left' or 'right')
    :param X: The features matrix.
    :param y: The class labels vector.
    """
    # Make the pytables table:
    f = h5py.File(outfile, mode='w')
    label = f.create_dataset('label', y.shape)
    set_name = f.create_dataset(feature_name, X.shape)

    # Load the data into it:
    set_name[...] = X
    label[...] = y

    # Save this new dataset to file
    f.flush()
    f.close()


def make_lr_fuel_file(outfile, inda, indb, indc, X, y):
    """
    Makes a FUEL dataset that combines both left and right features.
    :param outfile:
    :param inda:
    :param indb:
    :param indc:
    :param X:
    :param y:
    :return:
    """
    # Make the pytables table:
    f = h5py.File(outfile, mode='w')
    targets = f.create_dataset('targets', y.shape, dtype='int8')
    l_features = f.create_dataset('l_features', X['l'].shape, dtype='int8')
    r_features = f.create_dataset('r_features', X['r'].shape, dtype='int8')

    # Load the data into it:
    l_features[...] = X['l']
    r_features[...] = X['r']
    targets[...] = y

    # Label the axis:
    targets.dims[0].label = 'sample'
    targets.dims[1].label = 'class'
    l_features.dims[0].label = 'sample'
    l_features.dims[1].label = 'feature'
    r_features.dims[0].label = 'sample'
    r_features.dims[1].label = 'feature'

    # Make a "splits" dictionary as required by Fuel
    split_dict = {
        'train': {'l_features': (0, inda),
                  'r_features': (0, inda),
                  'targets': (0, inda)},
        'valid': {'l_features': (inda, inda + indb),
                  'r_features': (inda, inda + indb),
                  'targets': (inda, inda + indb)},
        'test': {'l_features': (inda + indb, inda + indb + indc),
                 'r_features': (inda + indb, inda + indb + indc),
                 'targets': (inda + indb, inda + indb + indc)},
    }

    f.attrs['split'] = H5PYDataset.create_split_array(split_dict)

    # Save this new dataset to file
    f.flush()
    f.close()


def make_one_sided_fuel_file(outfile, inda, indb, indc, X, y, side):
    """
    Makes a dataset that includes only a single side of features.
    :param outfile:
    :param inda:
    :param indb:
    :param indc:
    :param X:
    :param y:
    :param side:
    :return:
    """
    # Make the pytables table:
    f = h5py.File(outfile, mode='w')
    targets = f.create_dataset('targets', y.shape, dtype='int8')
    features = f.create_dataset('{}_features'.format(side), X.shape, dtype='int8')

    # Load the data into it:
    features[...] = X
    targets[...] = y

    # Label the axis:
    targets.dims[0].label = 'sample'
    targets.dims[1].label = 'class'
    features.dims[0].label = 'sample'
    features.dims[1].label = 'feature'

    # Make a "splits" dictionary as required by Fuel
    split_dict = {
        'train': {'{}_features'.format(side): (0, inda),
                  'targets': (0, inda)},
        'valid': {'{}_features'.format(side): (inda, inda + indb),
                  'targets': (inda, inda + indb)},
        'test': {'{}_features'.format(side): (inda + indb, inda + indb + indc),
                 'targets': (inda + indb, inda + indb + indc)},
    }

    f.attrs['split'] = H5PYDataset.create_split_array(split_dict)

    # Save this new dataset to file
    f.flush()
    f.close()