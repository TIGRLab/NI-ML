#!/bin/sh
# Usage: ./train.sh <model path>
if [ -z "$1" ];
then
    echo "Usage: ./train.sh <model path>";
    exit 1;
fi

WORKING=`pwd`
MODEL=$1
cd $MODEL
caffe train --solver=solver.prototxt --log_dir=./logs/
cd $WORKING

