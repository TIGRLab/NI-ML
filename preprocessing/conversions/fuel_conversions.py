from fuel.datasets import H5PYDataset
import h5py


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