#!/bin/bash
# Usage: ./clone_net_configs.sh <net type (L_HC/R_HC/CT/LR_HC/HC_CT)>
if [ $# -ne 1 ];
then
   echo "Usage: ./clone_net_configs.sh <net type (L_HC/R_HC/CT/HC/HC_CT)>";
   exit 1;
fi

#Copies the net_prototxt file from Fold_1 to Fold_{2..10} and then search and replaces variable names accordingly...
MODEL=$1

for (( i=2; i<=10; i++ ))
do
	cp OuterFold_1/net_partition_${MODEL}.prototxt OuterFold_${i}/net_partition_${MODEL}.prototxt
done

sed -i 's/Fold_1/Fold_2/g' OuterFold_2/net_partition_${MODEL}.prototxt
sed -i 's/Fold_1/Fold_3/g' OuterFold_3/net_partition_${MODEL}.prototxt
sed -i 's/Fold_1/Fold_4/g' OuterFold_4/net_partition_${MODEL}.prototxt
sed -i 's/Fold_1/Fold_5/g' OuterFold_5/net_partition_${MODEL}.prototxt
sed -i 's/Fold_1/Fold_6/g' OuterFold_6/net_partition_${MODEL}.prototxt
sed -i 's/Fold_1/Fold_7/g' OuterFold_7/net_partition_${MODEL}.prototxt
sed -i 's/Fold_1/Fold_8/g' OuterFold_8/net_partition_${MODEL}.prototxt
sed -i 's/Fold_1/Fold_9/g' OuterFold_9/net_partition_${MODEL}.prototxt
sed -i 's/Fold_1/Fold_10/g' OuterFold_10/net_partition_${MODEL}.prototxt
