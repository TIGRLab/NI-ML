#!/bin/bash -l
# separate the masks into left and right hippocampus
#
# Usage: me source.mnc output_stem [reg_l.xfm reg_r.xfm]
set -e 
module load minc-toolkit
module load ANTS

input_image=$1
output_stem=$2
out_l=${output_stem}_l.mnc
out_r=${output_stem}_r.mnc
reg_l=${3:-${output_stem}_l_Affine.txt}
reg_r=${4:-${output_stem}_r_Affine.txt}
out_l_std=${output_stem}_l_std.mnc
out_r_std=${output_stem}_r_std.mnc

std_l=02_std/model_labels_res_l_dist.mnc
std_r=02_std/model_labels_res_r_dist.mnc

lut_r="1 1; 2 1; 4 1; 5 1; 6 1;"
lut_l="101 1; 102 1; 104 1; 105 1; 106 1;"

#echo "
#input image: ${input_image}
#left HC: ${out_l}
#right HC: ${out_r}
#registration (left): ${reg_l}
#registration (right): ${reg_r}
#left HC standard space: ${out_l_std}
#right HC standard space: ${out_r_std}
#
#model (left): ${std_l}
#model (right): ${std_r}
#"

function ants_reg {
  fixed=$1
  moving=$2
  output_naming=$3
  ANTS 3 \
      -m MI[${fixed},${moving},1,32] \
      -o ${output_naming} \
      -i 0 \
      --use-Histogram-Matching \
      --number-of-affine-iterations 1000x1000x1000x1000x1000 \
      --rigid-affine true \
      --affine-gradient-descent-option 0.5x0.95x1.e-4x1.e-4 > /dev/null
}

function ants_transform {
    input=$1
    reference=$2
    transform=$3
    outputname=$4

    antsApplyTransforms \
        -i $input \
        -r $reference \
        -t $transform \
        -o $outputname \
        -n NearestNeighbor \
        --float > /dev/null
}

[ -e $out_l ] || minclookup -quiet -discrete -lut_string "${lut_l}" $input_image $out_l
[ -e $out_r ] || minclookup -quiet -discrete -lut_string "${lut_r}" $input_image $out_r

[ -e $reg_l ] || (mincmorph -distance $out_l ${out_l}_dist.mnc && ants_reg $std_l ${out_l}_dist.mnc ${output_stem}_l_ && rm ${out_l}_dist.mnc)
[ -e $reg_r ] || (mincmorph -distance $out_r ${out_r}_dist.mnc && ants_reg $std_r ${out_r}_dist.mnc ${output_stem}_r_ && rm ${out_r}_dist.mnc)

[ -e $out_l_std ] || (ants_transform $out_l $std_l $reg_l ${out_l_std}.tmp.mnc && mincresample -quiet -like ${std_l} -byte ${out_l_std}.tmp.mnc ${out_l_std} && rm ${out_l_std}.tmp.mnc)
[ -e $out_r_std ] || (ants_transform $out_r $std_r $reg_r ${out_r_std}.tmp.mnc && mincresample -quiet -like ${std_r} -byte ${out_r_std}.tmp.mnc ${out_r_std} && rm ${out_r_std}.tmp.mnc)
