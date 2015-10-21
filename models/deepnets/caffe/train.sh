#!/bin/sh
# Usage: ./train.sh <model path> [<gpu> <pretrained model file>]
if [ $# -lt 2 ];
then
    echo "Usage: ./train.sh <model path> [<gpu> <pretrained model file>]";
    exit 1;
fi

WORKING=`pwd`
MODEL=$1
GPU=-1
LOGS=./logs/
TRAIN=./train/
cd $MODEL

if [ ! -d "$LOGS" ];
then
    mkdir $LOGS;
fi

if [ ! -d "$TRAIN" ];
then
    mkdir $TRAIN;
fi

# TODO: Add support for specifying GPU and weights independently
if [ -n "$3" ]
then
  WEIGHTS=$3
  GPU=$2
  caffe train --solver=solver.prototxt -log_dir=$LOGS -weights $WEIGHTS;
else
  caffe train --solver=solver.prototxt -log_dir=$LOGS;
fi

cd $WORKING

