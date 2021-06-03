#!/usr/bin/env python
#coding=utf-8
"""Create and test completely sorted index (CSI) for specified column.

Syntax:
  h5index.py options h5file:/table

Options:
  -h  print this message.
  -c  name of column to be indexed.
  -t  name of column to test.

"""
import sys
import tables
import numpy as np
from time import time
from getopt import gnu_getopt
from signal import signal, SIGINT

def create_index(h5path, colname):
    h5file, h5node = h5path.split(':')
    assert tables.is_pytables_file(h5file)
    with tables.open_file(h5file, 'a') as h5:
        tab = h5.get_node(h5node)
        sys.stdout.write(u'    Creating completely sorted index (CSI) for column {} of {}:{}......'.format(colname, h5file, h5node))
        sys.stdout.flush()
        tab.cols.__getattribute__(colname).create_csindex()
        sys.stdout.write(u'\r    Creating completely sorted index (CSI) for column {} of {}:{}......OK.\n'.format(colname, h5file, h5node))
        sys.stdout.flush()

def test_index(h5path, colname):
    h5file, h5node = h5path.split(':')
    assert tables.is_pytables_file(h5file)
    with tables.open_file(h5file, 'r') as h5:
        tab = h5.get_node(h5node)
        assert getattr(tab.cols, colname).is_indexed, 'column `'+colname+'` is not indexed.'
        print('`{}` has been indexed.'.format(colname))
        print('Random I/O performance test (Press Ctrl+C to abort):')
        buf = np.empty((1,), dtype=tab.dtype)
        t = 0
        tic = time()
        while True:
            pos = int(np.random.rand(1)*tab.nrows)
            buf[:] = tab.read_sorted(colname, start=pos, stop=pos+1)[:]
            t += 1
            sys.stdout.write(u'\r  {:d} rows copied ({:.1f} Rows/s, {:.2f} KiB/s)......'.format(t, t/(time()-tic), 1e-3*tab.rowsize*t/(time()-tic)))
            sys.stdout.flush()

def handler(signal_rcvd, frame):
    print('\nAbort. Goodbye!')
    sys.exit(0)

if __name__ == '__main__':
    signal(SIGINT, handler)
    opts, args = gnu_getopt(sys.argv[1:], 'hc:t:')
    for opt, val in opts:
        if opt == '-h':
            print(__doc__)
            sys.exit()
        elif opt == '-c':
            colname = val
            h5path = args[0]
            create_index(h5path, colname)
        elif opt == '-t':
            colname = val
            h5path = args[0]
            test_index(h5path, colname)
