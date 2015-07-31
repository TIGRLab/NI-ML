# MODELS
The deepnets folder contains neural network models implemented in Caffe, Blocks, and Theano. See the README.md for details.

The other folders contain scripts for running non-MM models. See the python files contained therein for details. The config.json files are hyperparameter search specs used by the Spearmint Optimization library.

## Spearmint
The scripts in the AdaBoost, LR, and SVM folders are compatible with the Spearmint optimization library: they contain a config.json file that specifies hyperparameters to optimize for, and the .py script files each contain a main() function that returns the trained classifier's final loss on the test set.

Spearmint can be used on these models by running:
./run_spearmint.sh <model folder>

Spearmint will create an output folder in the model folder that will contain logs of each of the experiments run for that model.

If the config.json file for any of these models is edited (adding or removing a hyperparameter for Spearmint to optimize) then run ./clean_spearmint.sh <model folder> to remove any previous experiments, and to clear the DB of the current hyperparam search state.
 
