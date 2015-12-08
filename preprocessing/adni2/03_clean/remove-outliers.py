#!/usr/bin/env python
# vim: sw=4 ts=4 expandtab:
"""
Reads label data, and removes outliers. Stores the results in HDF5

Outlier labels are those that differ significantly from the mask (the average
hippocampus generated from ADNI1 data) that was used to create the bounding box
region when importing labels into hdf5.

Usage: 
    remove-outliers.py [options] <data.h5> <output.h5>

Options:
    --progress-freq N     Show progress every N images processed [default: 500]
"""

from docopt import docopt

def progress(complete,total,duration,period):
    print "Completed {}/{}. Last {} in {:.0f}s. Estimated time left: {:.0f}m".format(
            complete,total,period,duration,(duration/period)*(total-complete)/60)

if __name__ == '__main__':

    arguments = docopt(__doc__) 
    rawdatafile = arguments['<data.h5>']
    datafile = arguments['<output.h5>']
    printfreq = int(arguments['--progress-freq'])
    
    print "Loading libraries..."
    from time import time
    import glob
    import numpy as np
    import numpy.ma as ma
    import tables as tb
    import os.path

    FILTERS = tb.Filters(complevel=5,complib='zlib')
    rawfileh = tb.open_file(rawdatafile, mode='r', title="data", filters=FILTERS)
    rawdata = rawfileh.root.data

    labelfiles = rawfileh.root.files
    mask = rawfileh.root.mask

    print "Images loaded: ", len(labelfiles)
    print 


    #############################################################################
    # Compute scores
    # 
    # Do a pass over the data files, and compute the "score" (number of label
    # voxels different from the average mask)

    print "Computing mask difference scores..."
    scores = np.zeros(shape=(len(labelfiles)))  

    t0 = time()
    for i, labelfile in enumerate(labelfiles):
        if i>0 and i % printfreq == 0: 
            progress(i,len(labelfiles),time()-t0,printfreq)
            t0 = time()
    
        scores[i] = np.sum(np.abs(rawdata[i,:] - mask),axis=None)

    assert rawdata.shape[0] == scores.shape[0], "missing scores... "

    # exclude labels with scores differing too much from the mean
    passing_idx = scores < (np.average(scores) + np.std(scores))
    print "# labels with passing scores:", np.sum(passing_idx)

    ###########################################################################
    # Build data array
    # 
    # Ignore all high-scoring labels and labels with outlier voxels
    #
    # Read all data files, but only voxels in data mask. 

    print "Loading data into {}".format(datafile)
    fileh = tb.open_file(datafile, mode='w', title="data", filters=FILTERS)
    dataarray = fileh.createEArray(fileh.root,'data',tb.BoolAtom(), shape=[0] + list(mask.shape))

    t0 = time()
    images_used = []
    for i, labelfile in enumerate(labelfiles):
        if i>0 and i % printfreq == 0: 
            progress(i,len(labelfiles),time()-t0,printfreq)
            t0 = time()

        if not passing_idx[i]: continue

        data = rawdata[i,:]

        dataarray.append(data[np.newaxis,:])
        images_used.append(labelfile)

    _ = fileh.create_array(fileh.root,'files', images_used)
    _ = fileh.create_array(fileh.root,'fullmask', rawfileh.root.fullmask.read())
    _ = fileh.create_array(fileh.root,'mask', rawfileh.root.mask.read())
    _ = fileh.create_array(fileh.root,'cropbbox_min', rawfileh.root.cropbbox_min.read()) 
    _ = fileh.create_array(fileh.root,'cropbbox_max', rawfileh.root.cropbbox_max.read()) 
