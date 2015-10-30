#!/bin/bash
# Usage: ./process_caffe_logs.sh <net type (L_HC/R_HC/CT/LR_HC/HC_CT)> <tid> <start_fold> <fin_fold> <snapshot_iteration>
if [ $# -ne 5 ];
then
   echo "Usage: ./process_caffe_logs.sh <net type (L_HC/R_HC/CT/HC/HC_CT)> <trial_id> <start_fold> <fin_fold> <snapshot_iteration>";
   exit 1;
fi

MODEL=$1
TID=$2
S_FOLD=$3
F_FOLD=$4
ITER=$5

echo "Parsing train and test logs..."
for (( i=$S_FOLD; i<=$F_FOLD; i++ ))
do
	cd OuterFold_${i}; 
	sudo ~/caffe/tools/extra/parse_log.py log/caffe.INFO log --log_index ff_OF${i}_${MODEL}_T${TID}; 
	cd ..; 
done

echo "Copying parsed logs to /projects/nikhil/ADNI_prediction/caffe_output..."
for (( i=$S_FOLD; i<=$F_FOLD; i++ ))
do 
	sudo cp OuterFold_${i}/log/caffe.INFO.{train,test}_ff_OF${i}_${MODEL}_T${TID} /projects/nikhil/ADNI_prediction/caffe_output/OuterFold${i}; 
done

echo "Copying learned model to /projects/nikhil/ADNI_prediction/caffe_output..."
for (( i=$S_FOLD; i<=$F_FOLD; i++ ))
do 
	cp OuterFold_${i}/train/_iter_${ITER}.caffemodel /projects/nikhil/ADNI_prediction/caffe_output/OuterFold${i}/_iter_${ITER}.caffemodel_${MODEL}_T${TID}; 
done

echo "Copying model definition net.prototxt to /projects/nikhil/ADNI_prediction/caffe_output..."
for (( i=$S_FOLD; i<=$F_FOLD; i++ ))
do 
	cp OuterFold_${i}/net_partition.prototxt /projects/nikhil/ADNI_prediction/caffe_output/OuterFold${i}/net_partition.prototxt_${MODEL}_T${TID}; 
done

echo "fin!"

