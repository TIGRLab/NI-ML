# Blocks Models
This folder contains scripts to define and train Theano/Blocks neural network models. The
ff (feedforward) folder contains a model network that can be trained on features from
any structure by specifying the appropriate config file from the command line.

## Folder Contents
'ff' contains feed-forward models. The 'configs' and 'models' sub-folders have parameter
files and saved trained models respectively. The 'output' folder has log files produced
by the Spearmint optimization library whenever its used to train the models in 'ff'.


The config.json file specifies the hyperparameters used by Spearmint to optimize model
training. Finally, the ffnet.py file is the python script that defines and trains
the feedforward neural network models.

## Blocks Data
Blocks uses the FUEL data framework to provide input data for its models. FUEL
is a wrapper around h5, and uses a few simple conventions:
- Each data file can contain as many separate data nodes as needed, provided that
the training, validation, and test sets are all concatenated into the same file
and that the size of each split is encoded into the file.


The preprocessing/conversions/standardized_conversion.py script can produce FUEL
files for use with Blocks models. See the script for usage details.
