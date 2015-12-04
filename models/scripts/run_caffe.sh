#!/bin/sh
# Usage: ./run_caffe.sh <net type (L_HC/R_HC/CT/LR_HC/HC_CT)>
if [ $# -ne 1 ];
then
   echo "Usage: ./run_caffe.sh <net type (L_HC/R_HC/CT/HC/HC_CT)>";
   exit 1;
fi

MODEL=$1
# cp specific model name file to generic model name file to keep it consistent with solver.prototxt naming
cp $PWD/net_partition_${MODEL}.prototxt $PWD/net_partition.prototxt

MODEL_FILE=$PWD/net_partition.prototxt
SOLVER_FILE=$PWD/solver.prototxt

caffe train -gpu all -model $MODEL_FILE -solver $SOLVER_FILE --log_dir $PWD/log/
#caffe train -gpu all -model $MODEL_FILE -solver $SOLVER_FILE > $PWD/log/caffe.INFO 2>&1
