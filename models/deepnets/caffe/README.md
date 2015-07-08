# Caffe Models
This folder contains specifications for Caffe neural network models. The ff
(feedforward) model is split into structures, (EC, HC), and each structure split
into a left and right folder.

## Folder Contents
Caffe models are specified using a net.prototxt file that describes the network
architecture, plus a solver.prototxt file that descbribes the learning algorithm
and its hyperparameters.


In addition, each model folder contains .txt files that list the data files
that serve as input to the models: one for training, and one for testing/validation.


There are also train folders to contain saved model parameters, and logs folder to save
output to. Caffe saves models and outputs to these folders automatically.


The train.sh script can be used to start training models.

 ## Caffe Data
 Caffe data layers using h5 format require that the h5 file contains a node
 with the same name as the 'top' parameter of the data layer as specified in the
 net.prototxt file. ie. If the data layer in net.protoxt outputs 'l_ec_features',
 then Caffe expects the source data file to have a '/l_ec_features' node.


 The preprocessing/conversion/standardized_conversion.py script will produce
 Caffe-ready h5 files using this convention:


 Features nodes: <side>_<structure>_features ie. 'r_hc_features'
 Class label nodes: 'labels'


 The script outputs separate files for each structure, side, training, validation, and test
 sets as required by Caffe. See standardized_conversion.py for usage.


