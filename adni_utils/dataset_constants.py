"""
Constants used by the baseline models to load data when running experiments.
"""

# Data Vars:
source_path = '/projects/francisco/data/caffe/standardized/combined/'
# Sides to iterate over;
# ie: sides  = ['l', 'r', 'b']
sides = ['b']
structure = 'hc'

# Datasets to iterate over:
# ie. datasets = ['mci_cn', 'ad_cn', 'ad_mci_cn']
adni_datasets = ['balanced_ad_mci_cn']

# Use Fused only (otherwise use candidate segmentations)
use_fused = True
balance = True

# List of fold filename extensions to iterate over:
# ie. folds = ['_T1', '_T2', '_T3']
folds = [''] # No folds.

# Change these when running Spearmint experiments which use only the main and don't iterate over datasets or sides:
default_side = 'l' # Valid values: l, r, b
default_dataset = 'balanced_ad_mci_cn' # Valid values: mci_cn, ad_cn
