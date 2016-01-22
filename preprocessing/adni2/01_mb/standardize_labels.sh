#!/bin/bash -l
# separate the masks into left and right hippocampus
#
# Usage: me <tarball> <output-tarball>

input_tarball=$1
output_tarball=$2

module load GNU_PARALLEL
module load minc-toolkit/1.0.07

left_labels_lut="1 1; 2 2; 4 4; 5 5; 6 6;"
right_labels_lut="101 1; 102 2; 104 4; 105 5; 106 6;"
parallel="parallel"
stdlabel=std/ADNI_037_S_0150_MR_MPR-R__GradWarp__N3__Scaled_Br_20070806150947680_S11434_I65130.mnc

rm -rf /tmp/*

tempdir1=$(mktemp -d)
tempdir=$(mktemp -d)

tar -C $tempdir1 -xf $input_tarball
find $tempdir1 -name '*.mnc' -exec mv {} $tempdir \;
rm -rf $tempdir1

# extract left and right
output_dir=$tempdir/data/labels/left
mkdir ${output_dir} -p
${parallel} minclookup -quiet -discrete -lut_string \""${left_labels_lut}"\" {1} $output_dir/{1/} ::: $tempdir/*.mnc

output_dir=$tempdir/data/labels/right
mkdir ${output_dir} -p
${parallel} minclookup -quiet -discrete -lut_string \""${right_labels_lut}"\" {1} $output_dir/{1/} ::: $tempdir/*.mnc

# flip right
output_dir=$tempdir/data/labels/right/flipped
mkdir ${output_dir} -p
${parallel} bin/volflip {1} $output_dir/{1/} ::: $tempdir/data/labels/right/*.mnc

# lsq6 to a standard space
output_dir=$tempdir/data/labels/std
mkdir ${output_dir} -p
(
${parallel} echo bestlinreg.pl -lsq6 {1} ${stdlabel} $output_dir/{1/.}_right.xfm $output_dir/{1/.}_right.mnc ::: $tempdir/data/labels/right/flipped/*.mnc
${parallel} echo bestlinreg.pl -lsq6 {1} ${stdlabel} $output_dir/{1/.}_left.xfm  $output_dir/{1/.}_left.mnc ::: $tempdir/data/labels/left/*.mnc
) | $parallel

( cd $tempdir/data/labels/std && tar -czf $output_tarball *.mnc )

rm -rf $tempdir 
