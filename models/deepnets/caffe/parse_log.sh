#!/bin/bash
#
# Plots the current/last run Caffe net at given path.
#
# Usage: ./parse_log.sh <path to log file>
# Requires a $CAFFE root folder env variable to be set due to dependencies.
export CAFFE=/home/m/mchakrav/nikhil/scratch/deep_learning/caffe/
if [ -z "$1" ];
then
    echo "Usage: ./parse_log.sh <log file>";
    exit 1;
fi

if [ -z "$CAFFE" ];
then
   echo "Set CAFFE variable to point to the root of your caffe repo";
   exit 1;
fi

WORKING=`pwd`
LOG_FILE=$1
LOG_DIR="$(dirname "$LOG_FILE")"
echo $LOG_DIR

python $CAFFE/tools/extra/parse_log.py --verbose $LOG_FILE $LOG_DIR/

ret=$?;

if [[ $ret != 0 ]];
then
  echo "Log cannot be parsed. Maybe not enough iterations yet?";
  exit 1;
fi

#python ../../../visualizations/caffe_plots/plot_caffe_logs.py $MODEL/logs/
