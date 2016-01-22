#!/usr/bin/env python
# vim: sw=4 ts=4 expandtab:
"""
Apply the mask from another dataset.

Usage: 
    remask.py [options] <source.h5> <like.h5> <target.h5>

Options:
    --left
    --progress-freq N     Show progress every N images processed [default: 10]
"""

from time import time
import datetime
import glob
import itertools
import math
import matplotlib
import numpy as np
import tables as tb
import os.path
import random
import sys
from docopt import docopt

def progress(complete,total,duration,period):
    print "Completed {}/{}. Last {} in {:.0f}s. Estimated time left: {:.0f}m".format(
            complete,total,period,duration,(duration/period)*(total-complete)/60)

if __name__ == '__main__':

    arguments = docopt(__doc__) 
    source    = arguments['<source.h5>']
    like      = arguments['<like.h5>']
    targetfile= arguments['<target.h5>']
    printfreq = int(arguments['--progress-freq'])

    FILTERS = tb.Filters(complevel=5,complib='zlib')
    sourceh = tb.open_file(source  , mode='r', filters=FILTERS)
    likeh   = tb.open_file(like    , mode='r', filters=FILTERS)
    targeth = tb.open_file(targetfile, mode='w', filters=FILTERS)

    source_mask = sourceh.root.datamask
    source_data = sourceh.root.data
    like_mask   = likeh.root.datamask
    like_data   = likeh.root.data

    target_data = targeth.createCArray(
                    targeth.root,'data',tb.Int8Atom(), 
                    shape=[source_data.shape[0]]+list(like_data.shape[1:]))

    # copy data
    n_images = source_data.shape[0]
    print "Remasking {} images...".format(n_images)
    t0 = time()
    for i in range(n_images):
        if i>0 and i % printfreq == 0: 
            progress(i,n_images,time()-t0,printfreq)
            t0 = time()
        reconstituted              = np.zeros(shape=source_mask.shape)
        reconstituted[np.array(source_mask)] = source_data[i,:]
        target_data[i,:]           = reconstituted[np.array(like_mask)] 
    
    _ = targeth.create_array(targeth.root,'fullvolmask' ,likeh.root.fullvolmask.read())
    _ = targeth.create_array(targeth.root,'volmask'     ,likeh.root.volmask.read())
    _ = targeth.create_array(targeth.root,'datamask'    ,likeh.root.datamask.read())
    _ = targeth.create_array(targeth.root,'cropbbox_min',likeh.root.cropbbox_min.read()) 
    _ = targeth.create_array(targeth.root,'cropbbox_max',likeh.root.cropbbox_max.read()) 

    sourceh.close()
    likeh.close()
    targeth.close()
