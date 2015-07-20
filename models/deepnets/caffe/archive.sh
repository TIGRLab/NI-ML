#!/bin/sh
# Usage: ./archive.sh <model path> <iteration>
if [ $# -ne 2 ];
then
   echo "Usage: ./archive.sh <model path> <iteration number>";
   exit 1;
fi

MODEL=$1
ITER=$2
DATE=`date +%Y-%m-%d`

MODEL_FILE=$MODEL/train/_iter_$ITER.caffemodel
SOLVER_FILE=$MODEL/train/_iter_$ITER.solverstate

cp $MODEL_FILE $MODEL/archive/$DATE.$ITER.caffemodel
cp $SOLVER_FILE $MODEL/archive/$DATE.$ITER.solverstate
cp $MODEL/logs/caffe.INFO $MODEL/archive/$DATE.INFO
cp $MODEL/net.prototxt $MODEL/archive/$DATE.net.prototxt
cp $MODEL/solver.prototxt $MODEL/archive/$DATE.solver.prototxt

echo "Archived with $DATE timestamp"
