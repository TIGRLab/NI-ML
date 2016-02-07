#!/usr/bin/env python
# vim: sw=4 ts=4 expandtab:
"""
Reads a bunch of label files and stores in HDF5 format. 

Usage: 
    hd5ify.py <mask.mnc> <hfile.h5> <imagedir> 
"""

from docopt import docopt

# voxel padding around mask bounding box
PADDING = 5  

if __name__ == '__main__':

    arguments = docopt(__doc__) 
    maskfile = arguments['<mask.mnc>']
    imagedir = arguments['<imagedir>']
    datafile = arguments['<hfile.h5>']

    from pyminc.volumes.factory import *
    import glob
    import numpy as np
    import tables as tb
    import os.path

    labelfiles = glob.glob(imagedir + '/*.mnc')

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
    minidx = maskidx.min(0) - PADDING
    maxidx = maskidx.max(0) + PADDING
    mask = fullmask[minidx[0]:maxidx[0],minidx[1]:maxidx[1],minidx[2]:maxidx[2]]

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
        files  = fileh.create_array(fileh.root,'files',labelfiles)
    else: 
        files  = fileh.root.files[:]
        fileh.remove_node(fileh.root.files)
        _ = fileh.create_array(fileh.root,'files',files + labelfiles)

    if not 'fullmask' in fileh.root:
        _ = fileh.create_array(fileh.root,'fullmask',fullmask)
    if not 'mask' in fileh.root:
        _ = fileh.create_array(fileh.root,'mask',mask)
    if not 'cropbbox_min' in fileh.root:
        _ = fileh.create_array(fileh.root,'cropbbox_min', minidx) 
    if not 'cropbbox_max' in fileh.root:
        _ = fileh.create_array(fileh.root,'cropbbox_max', maxidx) 
    if not 'data' in fileh.root:
        dataarray = fileh.createEArray(fileh.root,'data',tb.BoolAtom(), 
                shape=[0]+list(mask.shape))
    else: 
        dataarray = fileh.root.data

    for i, labelfile in enumerate(labelfiles):
        data = volumeFromFile(labelfile).data[
            minidx[0]:maxidx[0],
            minidx[1]:maxidx[1],
            minidx[2]:maxidx[2]] > 0

        dataarray.append(data[np.newaxis,:])

    fileh.close()
