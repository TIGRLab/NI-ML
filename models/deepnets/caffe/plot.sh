#!/bin/bash
#
# Plots the current/last run Caffe net at given path.
#
# Usage: ./plot.sh <path to model>
# Requires a $CAFFE root folder env variable to be set due to dependencies.
export CAFFE=/projects/francisco/repositories/caffe/
if [ -z "$1" ];
then
    echo "Usage: ./train.sh <model path>";
    exit 1;
fi

if [ -z "$CAFFE" ];
then
   echo "Set CAFFE variable to point to the root of your caffe repo";
   exit 1;
fi

WORKING=`pwd`
MODEL=$1
python $CAFFE/tools/extra/parse_log.py --verbose $MODEL/logs/caffe.INFO $MODEL/logs/

ret=$?;

if [[ $ret != 0 ]];
then
  echo "Log cannot be parsed. Maybe not enough iterations yet?";
  exit 1;
fi

python ../../../visualizations/caffe_plots/plot_caffe_logs.py $MODEL/logs/
