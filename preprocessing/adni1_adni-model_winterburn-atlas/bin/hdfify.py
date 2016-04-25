#!/usr/bin/env python
# vim: sw=4 ts=4 expandtab:
"""
Reads a bunch of label files and stores in HDF5 format. 

Usage: 
    hd5ify.py [options] <imagedir> <hfile.h5>

Options:
    --mask MASK           Mask for 1st stage filtering [default: average-mask.mnc]
    --progress-freq N     Show progress every N images processed [default: 500]
"""

from docopt import docopt

def progress(complete,total,duration,period):
    print "Completed {}/{}. Last {} in {:.0f}s. Estimated time left: {:.0f}m".format(
            complete,total,period,duration,(duration/period)*(total-complete)/60)

if __name__ == '__main__':

    from pyminc.volumes.factory import *

    arguments = docopt(__doc__) 
    maskfile = arguments['--mask']
    imagedir = arguments['<imagedir>']
    datafile = arguments['<hfile.h5>']
    printfreq         = int(arguments['--progress-freq'])

    print "Loading libraries..."
    from pyminc.volumes.factory import *
    from time import time
    import glob
    import numpy as np
    import numpy.ma as ma
    import tables as tb
    import os.path

    print "Creating list of images..."
    labelfiles = glob.glob(imagedir + '/*.mnc')

    print "{} images found.".format(len(labelfiles))
    print 

    #############################################################################
    # Load average mask and compute a bounding box
    # 
    # Because of memory limits, we can't read in a full volume from each image, So,
    # we'll take a stab at a goodly sized bounding box based on the average mask of
    # the voted labels

    # Get bounding box from average
    fullmask = volumeFromFile(maskfile).data

    # compute bounding box
    maskidx = np.argwhere(fullmask)
    minidx = maskidx.min(0) - 5
    maxidx = maskidx.max(0) + 5
    mask = fullmask[minidx[0]:maxidx[0],minidx[1]:maxidx[1],minidx[2]:maxidx[2]]

    # indices to extract at first
    print "Mask bounding box: ", minidx, maxidx
    print

    ###########################################################################
    # Build data array
    # 
    # Read all data files, but only voxels in data mask. 
    FILTERS = tb.Filters(complevel=5,complib='zlib')
    if os.path.exists(datafile):
        fileh = tb.open_file(datafile, mode='a', title="data", filters=FILTERS)
    else: 
        fileh = tb.open_file(datafile, mode='w', title="data", filters=FILTERS)

    if not 'files' in fileh.root:
        _ = fileh.create_array(fileh.root,'files',labelfiles)
    if not 'fullmask' in fileh.root:
        _ = fileh.create_array(fileh.root,'fullmask',fullmask)
    if not 'mask' in fileh.root:
        _ = fileh.create_array(fileh.root,'mask',mask)
    if not 'cropbbox_min' in fileh.root:
        _ = fileh.create_array(fileh.root,'cropbbox_min', minidx) 
    if not 'cropbbox_max' in fileh.root:
        _ = fileh.create_array(fileh.root,'cropbbox_max', maxidx) 
    if not 'data' in fileh.root:
        print "Loading data into {}".format(datafile)
        dataarray = fileh.createEArray(fileh.root,'data',tb.BoolAtom(), 
                shape=[0]+list(mask.shape))

        t0 = time()
        for i, labelfile in enumerate(labelfiles):
            if i>0 and i % printfreq == 0: 
                progress(i,len(labelfiles),time()-t0,printfreq)
                t0 = time()

            data = volumeFromFile(labelfile).data[
                minidx[0]:maxidx[0],
                minidx[1]:maxidx[1],
                minidx[2]:maxidx[2]] > 0

            dataarray.append(data[np.newaxis,:])
