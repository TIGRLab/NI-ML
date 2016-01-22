#!/bin/bash -l 
# Imports right and left labels into hdf5 files
# 
# Usage: hdfify_job.sh mask.mnc labelsdir/ left.h5 right.h5
set -e
set -x 

maskfile=$1
labelsdir=$2
left=$3
right=$4

tempdir=$(mktemp --tmpdir=/dev/shm -d)
parallel tar -C $tempdir -xf {} ::: $labelsdir/*.tar*

mkdir $tempdir/left $tempdir/right
find $tempdir -maxdepth 1 -name '*left.mnc'  | parallel mv {} $tempdir/left 
find $tempdir -maxdepth 1 -name '*right.mnc'  | parallel mv {} $tempdir/right 

./hdfify.py --mask $maskfile $tempdir/left  $left & 
./hdfify.py --mask $maskfile $tempdir/right $right &
wait

rm -rf $tempdir
