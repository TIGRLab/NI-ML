#!/usr/bin/env python
# vim: sw=4 ts=4 expandtab:
"""
Apply the mask from another dataset.

Usage: 
    remask.py [options] <source.h5> <like.h5> <target.h5>

Options:
    --progress-freq N     Show progress every N images processed [default: 10]
"""

from time import time
import glob
import itertools
import math
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

    source_data  = sourceh.root.data
    source_cropbbox_min = sourceh.root.cropbbox_min[:]
    source_cropbbox_max = sourceh.root.cropbbox_max[:]

    like_mask    = likeh.root.r_datamask[:]
    like_data    = likeh.root.r_test_data
    like_volmask = likeh.root.r_volmask[:]
    like_fullvolmask = likeh.root.r_fullvolmask[:]
    like_cropbbox_min = likeh.root.r_cropbbox_min[:]
    like_cropbbox_max = likeh.root.r_cropbbox_max[:]

    # move like-mask to source-mask cropbox: 
    # step 1: reconstitute the 3D datamask
    datamaskvol = np.zeros(like_volmask.shape).reshape(-1)
    datamaskvol[like_mask] = 1
    datamaskvol = datamaskvol.reshape(like_volmask.shape)

    # step 2: insert the 3D datamask into the whole volume
    wholevol = np.zeros(like_fullvolmask.shape)
    wholevol[
        like_cropbbox_min[0]:like_cropbbox_max[0],
        like_cropbbox_min[1]:like_cropbbox_max[1],
        like_cropbbox_min[2]:like_cropbbox_max[2]] = datamaskvol

    # step 3: extract the datamask using the ADNI2 cropbbox
    new_mask = wholevol[
            source_cropbbox_min[0]:source_cropbbox_max[0],
            source_cropbbox_min[1]:source_cropbbox_max[1],
            source_cropbbox_min[2]:source_cropbbox_max[2]].astype('bool')
    new_mask_flat = new_mask.reshape(-1)
    
    # copy data
    target_data = targeth.createCArray(
                    targeth.root,'data',tb.Int8Atom(), 
                    shape=[source_data.shape[0], like_data.shape[1]])

    _ = targeth.create_array(targeth.root,'files'       ,sourceh.root.files.read())
    _ = targeth.create_array(targeth.root,'fullvolmask' ,like_fullvolmask)
    _ = targeth.create_array(targeth.root,'datamask'    ,new_mask)
    _ = targeth.create_array(targeth.root,'cropbbox_min',source_cropbbox_min) 
    _ = targeth.create_array(targeth.root,'cropbbox_max',source_cropbbox_max) 

    blocksize = 8000
    n_images = source_data.shape[0]
    print "Masking {} images...".format(n_images)
    for i in range(0, n_images, blocksize):
        print "masking images",i,'through',i+blocksize,'of',n_images
        start = i
        stop = min(n_images,i+blocksize)
        block = source_data[start:stop,:]
        flatblock = block.reshape(block.shape[0],-1)
        masked = flatblock[:,new_mask_flat]
        target_data[start:stop,:] = masked
        empty = np.argwhere(np.sum(masked, axis=1) == 0)
        if empty:
            print "empty slices:", empty
    

    sourceh.close()
    likeh.close()
    targeth.close()
