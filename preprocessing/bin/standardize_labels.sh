#!/bin/bash
# Register right/left HC labels to a standard space
#
# Usage: me <tarball> <output-tarball> <standard.mnc>
#
# Arguments:
#    <input.tar.gz>     A tarball of labels (will search within subfolders)
#    <output.tar.gz>    Path of file to output standardized labels to
#    <standard.mnc>     Label (must be left HC) to lsq6 standardize to

set -o errexit
set -o errtrace
set -o nounset
set -o pipefail

input_tarball=$1
output_tarball=$(readlink -f $2)

# Label to register all other labels to
# eg. std/ADNI_037_S_0150_MR_MPR-R__GradWarp__N3__Scaled_Br_20070806150947680_S11434_I65130.mnc
stdspace=$3

echo source: $input_tarball
echo target: $output_tarball


# lookup table that maps left labels (Winterburn atlas)
LEFT_LUT="1 1; 2 2; 4 4; 5 5; 6 6;"

# lookup table that maps right labels (winterburn atlas)
RIGHT_LUT="101 1; 102 2; 104 4; 105 5; 106 6;"

tempdir1=$(mktemp -d)
tempdir=$(mktemp -d)

tar -C $tempdir1 -xf $input_tarball
find $tempdir1 -name '*.mnc' -exec mv {} $tempdir \;
rm -rf $tempdir1

# extract left and right labels
output_dir=$tempdir/data/labels/left
mkdir ${output_dir} -p
parallel minclookup -quiet -discrete -lut_string \""${LEFT_LUT}"\" {1} $output_dir/{1/} ::: $tempdir/*.mnc

output_dir=$tempdir/data/labels/right
mkdir ${output_dir} -p
parallel minclookup -quiet -discrete -lut_string \""${RIGHT_LUT}"\" {1} $output_dir/{1/} ::: $tempdir/*.mnc

# flip right label 
output_dir=$tempdir/data/labels/right/flipped
mkdir ${output_dir} -p
parallel volflip {1} $output_dir/{1/} ::: $tempdir/data/labels/right/*.mnc

# lsq6 to a standard space
output_dir=$tempdir/data/labels/std
mkdir ${output_dir} ${output_dir}/left ${output_dir}/right -p
(
parallel echo bestlinreg.pl -lsq6 {1} ${stdspace} $output_dir/{1/.}_right.xfm $output_dir/right/{1/.}_right.mnc ::: $tempdir/data/labels/right/flipped/*.mnc
parallel echo bestlinreg.pl -lsq6 {1} ${stdspace} $output_dir/{1/.}_left.xfm  $output_dir/left/{1/.}_left.mnc ::: $tempdir/data/labels/left/*.mnc
) | parallel -v

( cd $tempdir/data/labels && tar -czf $output_tarball std/left std/right )

rm -rf $tempdir 
