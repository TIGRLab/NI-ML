"""
Converts dataset extracted from ADNI to a format that is usable for Caffe neural networks.

Usage:
    caffe_conversion.py <target_path>
"""

import logging
from docopt import docopt
import h5py
from adni_data import split_3_way, datafile, features_template, labels_template, sides, load_data


def make_file(outfile, data, X, y):
    """
    Create a Caffe-format HDf5 data and labels file.
    :param outfile: A path and filename to write the dataset out to.
    :param data: Name of the dataset (ie. 'left' or 'right')
    :param X: The features matrix.
    :param y: The class labels vector.
    """
    # Make the pytables table:
    f = h5py.File(outfile, mode='w')
    label = f.create_dataset('label', y.shape)
    data = f.create_dataset('{}_features'.format(data), X.shape)

    # Load the data into it:
    data[...] = X
    label[...] = y

    # Save this new dataset to file
    f.flush()
    f.close()


if __name__ == "__main__":
    arguments = docopt(__doc__)
    target_path = arguments['<target_path>']
    logging.basicConfig(level=logging.DEBUG)

    datah = load_data()

    X = []
    y = []

    for side in sides:
        for split in ['train', 'valid', 'test']:
            X = datah.get_node(features_template.format(side, split))[:]
            y = datah.get_node(labels_template.format(side, split))[:]

            # Split data into 3 sets for 2-way classification of each category pair.
            sets = split_3_way(X, y)
            for k, v in sets.items():
                logging.info('Making {} split for {} set'.format(split, k))
                make_file('{}{}_{}_{}.h5'.format(target_path, side, k, split), side, v['X'], v['y'])