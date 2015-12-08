#!/bin/bash
# usage: inputfile outputfile

unpackdir=$(mktemp --tmpdir=/export/ramdisk -d)
convdir=$(mktemp --tmpdir=/export/ramdisk -d)

tar -C $unpackdir -xf $1
parallel mincconvert -2 -compress 9 {} $convdir/{/} ::: $unpackdir/*.mnc  2>/dev/null
( cd $convdir && tar -czf $2 *.mnc )

rm -rf $unpackdir $convdir
