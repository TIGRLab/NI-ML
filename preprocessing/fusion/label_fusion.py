from fuel.datasets import H5PYDataset
import numpy as np
import h5py as h5
import collections
import scipy.stats as stats


def make_fused_set(features, labels, files, data_set, target_file):
    #Start fusing
    subject_idx = []
    subject_vol_dict = collections.defaultdict(list)
    subject_class_dict = collections.defaultdict(list)
    j=0
    #find volume indices for each unique subject
    for i in files:
        paresd_str = str(i).split('_')
        subject_id = int(paresd_str[4])
        subject_idx.append(subject_id)
        subject_vol_dict[subject_id].append(j)
        subject_class_dict[subject_id].append(labels[j])
        j=j+1

    print len(subject_vol_dict), len(subject_class_dict)
    fuse_vol_array = np.zeros((len(set(subject_idx)),features.shape[1]))
    fuse_class_array = np.zeros(len(set(subject_idx)))
    j=0
    for i in set(subject_idx):
        fuse_vol = stats.mode(features[subject_vol_dict[i]])
        fuse_vol_array[j,:] = fuse_vol[0]
        fuse_class_array[j] = stats.mode(subject_class_dict[i])[0]
        j=j+1

    #Save fused files
    output_data = h5.File(target_file, 'a')
    output_data.create_dataset('{}_data_fused'.format(data_set), data=fuse_vol_array)
    output_data.create_dataset('{}_class_fused'.format(data_set), data=fuse_class_array)
    output_data.close()


if __name__ == "__main__":
    right_dim = 10519
    left_dim = 11427

    #Import candidate labels and subject file
    output_file = '/projects/francisco/data/fused_test.h5'
    input_file = '/projects/jp/adni-autoencoder/combined.h5' #Fused data but need to get activations first
    input_data = h5.File(input_file, 'r')

    files = input_data['l_test_files'][:]
    labels = input_data['l_test_classes'][:]
    l_features = input_data['l_test_data'][:]
    r_features = input_data['r_test_data'][:]
    features = np.concatenate((l_features, r_features), axis=1)

    make_fused_set(features, labels, files, 'test', output_file)

