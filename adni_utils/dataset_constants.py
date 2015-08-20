import pandas as pd
from data import load_matrices

# Data Vars:
source_path = '/projects/francisco/data/caffe/standardized/combined/'

# Use Fused only (otherwise use candidate segmentations)
use_fused = True
balance = True

# List of fold filename extensions to iterate over:
# ie. folds = ['_T1', '_T2', '_T3']
default_folds = [''] # No folds.

# Change these when running Spearmint experiments which use only the main and don't iterate over datasets or sides:
default_dataset = 'ADNI_Cortical_Features' # Valid values: mci_cn, ad_cn

# How many trials to run per fold (useful in the case of randomly sampled subsets of data, or randomized algos):
default_n_trials = 10

class_name_map = {
    'ad_mci_cn': ['ad', 'cn', 'mci'],
    'ad_cn': ['ad', 'cn'],
    'mci_cn': ['cn', 'mci'],
    'ADNI_Cortical_Features': ['ad', 'cn', 'mci']
}

dataset_args = {
    'ADNI_Cortical_Features': {
        'source_path': source_path,
        'class_names': ['ad', 'cn', 'mci'],
        'load_fn': load_matrices,
        'dataset': 'ADNI_Cortical_Features',
        'omit_class': 2, # ie: Omit ad, classifies between cn and mci
        'use_fused': False
    },
    'HC': {
        'source_path': source_path,
        'class_names': ['ad', 'cn', 'mci'],
        'load_fn': load_matrices,
        'dataset': 'hc',
        'omit_class': None,
        'structure': 'hc',
        'sides': ['b'],
        'use_fused': True,
    },
}
