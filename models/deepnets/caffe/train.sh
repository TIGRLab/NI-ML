#!/bin/sh
# Usage: ./train.sh <model path> [<pretrained model file>]
if [ -z "$1" ];
then
    echo "Usage: ./train.sh <model path> [<pre-trained model file>]";
    exit 1;
fi


WORKING=`pwd`
MODEL=$1
cd $MODEL

if [ -n "$2" ]
then
  WEIGHTS=$2
  caffe train --solver=solver.prototxt --log_dir=./logs/ -weights $WEIGHTS -gpu 0;
else
  caffe train --solver=solver.prototxt --log_dir=./logs/ -gpu 0;
fi

cd $WORKING

