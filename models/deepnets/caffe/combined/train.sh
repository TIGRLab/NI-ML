#!/bin/sh
# Usage: ./train.sh <path_to_solver>
MODEL_DIR=`pwd`
LOG_DIR=`pwd`/../logs/
TARGET_DIR=$1
cd $1
caffe train --solver=solver.prototxt --log_dir=$(LOG_DIR)
cd $(MODEL_DIR)
echo "Log file: $(LOG_DIR)/caffe.INFO"
