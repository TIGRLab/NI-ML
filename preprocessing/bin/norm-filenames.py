#!/usr/bin/env python
# vim: sw=4 ts=4 expandtab:
"""
Normalizes the file names from adni1 and adni2 candidate labels
Also, adds a dataset that indicates adni1/adni2 membership

Usage: 
    norm-filenames.py [options] <data.h5>
"""

from docopt import docopt
import os.path
import sys

if __name__ == '__main__':

    arguments = docopt(__doc__) 
    datafile = arguments['<data.h5>']

    import tables as tb
    import os.path
    import re

    FILTERS = tb.Filters(complevel=5,complib='zlib')
    datah = tb.open_file(datafile, mode='a', title="data", filters=FILTERS)

    labelfiles = datah.root.files.read()
    adni1_or_not = [ 'subject' not in x for x in labelfiles]
    normedfiles = map(lambda x: re.sub(r"_\d+$","",x),
    map(lambda x: re.sub("subject.*\.","", x), 
    map(lambda x: re.sub('.*/','', x 
        .replace('_left','')
        .replace('_right','')
        .replace('_std','')
        .replace('_labels','')
        .replace('_label','')
        .replace('.mnc','')), 
        labelfiles)))
    
    _ = datah.create_array(datah.root, 'normedfiles', normedfiles)
    _ = datah.create_array(datah.root, 'adni1', adni1_or_not)
    datah.close()
