#!/bin/bash -l
# Copies all candidate labels into a single hdf5 file
#
# Usage:
#   me <left.h5> <right.h5> <tarfile>..
#

temp1=$(mktemp --tmpdir=/dev/shm -d tmp.XXXXXX)
temp2=$(mktemp --tmpdir=/dev/shm -d tmp.XXXXXX)

function finish { rm -rf $temp1 $temp2; }; trap finish EXIT

left=$1; shift;
right=$1; shift;

echo $left $right

module load minc-toolkit python python-extras pyminc

for tarfile in $@; do 
    echo $tarfile
    tar -C $temp1 -xf $tarfile

    mkdir $temp2/l_std; find $temp1 -name '*_l_std.mnc' -exec mv '{}' $temp2/l_std \;
    mkdir $temp2/r_std; find $temp1 -name '*_r_std.mnc' -exec mv '{}' $temp2/r_std \; 

    bin/hdfify.py 02_std/candidate_labels_bbox_l_std.mnc $left $temp2/l_std
    bin/hdfify.py 02_std/candidate_labels_bbox_r_std.mnc $right $temp2/r_std

    rm -rf $temp1/* $temp2/*;
done

