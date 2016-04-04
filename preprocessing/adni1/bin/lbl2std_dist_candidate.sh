#!/bin/bash -l
# separate the masks into left and right hippocampus
#
# Usage: me source.tar output_dir
set -e 

tempdir1=$(mktemp -d --tmpdir=/tmp tmp.XXXXXXXXXX)
tempdir=$(mktemp -d --tmpdir=/tmp tmp.XXXXXXXXXX)

function finish {
    rm -rf $tempdir1 $tempdir
}

trap finish EXIT

source_tar=$1
output_dir=$2
mkdir -p $output_dir
output_tar=$output_dir/$(basename $1)
output_stem=$(echo $source_tar | grep -o 'ADNI_.*I[0-9]*')
reg_l=02_std/fused_labels/${output_stem}_l_Affine.txt
reg_r=02_std/fused_labels/${output_stem}_r_Affine.txt

[ -e $output_tar ] && exit; 

if [ ! -e $reg_l -o ! -e $reg_r ]; then 
    echo "$reg_l or $reg_r does not exist. Skipping $source_tar."
    exit 1;
fi

tar -C $tempdir1 -xf $source_tar

find $tempdir1 -name '*.mnc' | while read label; do 
    stem=$(echo $label | grep -o 'ADNI.*I[0-9]*' | sed 's/\//./g')
    output_stem=$tempdir/$stem
    out_l=${output_stem}_l.mnc
    out_r=${output_stem}_r.mnc

    # do the minclookups here because the lut table is different than the fused
    # labels
    lut_r="2 1;"
    lut_l="1 1;"
    minclookup -quiet -discrete -lut_string "${lut_l}" $label $out_l
    minclookup -quiet -discrete -lut_string "${lut_r}" $label $out_r

    bin/lbl2std_dist.sh $label $output_stem $reg_l $reg_r
done

tar -czf $output_tar $tempdir
