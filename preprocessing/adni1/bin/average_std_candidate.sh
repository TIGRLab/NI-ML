#!/bin/bash -l
# Averages the left/right standardized labels in a candidate label tarfile
#
# Usage:
#   me <tarfile>
#
# Outputs: 
#   <tarfile stem>_l_std_avg.mnc
#   <tarfile stem>_r_std_avg.mnc

tarfile=$1
tarfile_stem=${tarfile/.tar*/}
tempfolder=$(mktemp --tmpdir=/tmp -d tmp.XXXXXX)

function finish {
    rm -rf $tempfolder
}

trap finish EXIT

echo $tarfile

tar -C $tempfolder -xf $tarfile

module load minc-toolkit

mincinfo ${tarfile_stem}_l_std.mnc >/dev/null 2>&1
[ $? -eq 0 ] || { rm -f ${tarfile_stem}_l_std.mnc; mincaverage $(find $tempfolder -name '*_l_std.mnc') ${tarfile_stem}_l_std.mnc; }

mincinfo ${tarfile_stem}_r_std.mnc >/dev/null 2>&1
[ $? -eq 0 ] || { rm -f ${tarfile_stem}_r_std.mnc; mincaverage $(find $tempfolder -name '*_r_std.mnc') ${tarfile_stem}_r_std.mnc; }

[ -e ${tarfile_stem}_l_std.mnc ] || mincaverage $(find $tempfolder -name '*_l_std.mnc') ${tarfile_stem}_l_std.mnc
[ -e ${tarfile_stem}_r_std.mnc ] || mincaverage $(find $tempfolder -name '*_r_std.mnc') ${tarfile_stem}_r_std.mnc
