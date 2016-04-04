#!/usr/bin/env python 
import h5py
import glob
import os.path
import numpy as np
import time
from multiprocessing import Process, Queue, JoinableQueue

CHUNKSIZE = 5

def enqueue_volumes(q, minc_volumes): 
    zeros = np.zeros((CHUNKSIZE,) + shape)
    for i in xrange(0, len(minc_volumes), CHUNKSIZE):
        names = []
        for j,v in enumerate(minc_volumes[i:i+CHUNKSIZE]):
            print v
            vf = h5py.File(v,'r')
            image = vf['minc-2.0']['image']['0']['image']
            zeros[j,:] = image
            names.append(os.path.basename(v))
            vf.close()
        q.put([i, j, zeros, names], False)
        q.join()
    q.put(None)

if __name__ == '__main__':
    minc_volumes = glob.glob('02_std/fused_labels/*_l_std.mnc')[:150]

    vf = h5py.File(minc_volumes[0],'r')
    image = vf['minc-2.0']['image']['0']['image']
    shape = image.shape


    f = h5py.File('/dev/shm/test.h5','w')
    data = f.create_dataset('data', (len(minc_volumes),) + image.shape, 
            chunks=(CHUNKSIZE,) + image.shape, compression='gzip')
    names = f.create_dataset('names', (len(minc_volumes),), 
            dtype=h5py.special_dtype(vlen=unicode))

    vf.close()

    q = JoinableQueue()
    p = Process(target=enqueue_volumes, args=(q,minc_volumes))
    p.start()

    while True: 
        item = q.get(block=True)
        q.task_done()

        if item is None: 
            break
        i, j, zeros, namelist = item

        print 'data[{}:{},:] = zeros[:{},:]'.format(i, i+j+1, j+1)
        data[i:i+j+1,:] = zeros[:j+1,:]
        print 'names[{}:{}] = namelist'.format(i, i+j+1)
        print "qsize:", q.qsize()
        names[i:i+j+1] = namelist

    f.close()
