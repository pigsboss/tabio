#!/usr/bin/env python3
#coding=utf-8
"""Select and save HDF5 table to specified container.

Syntax:
  h5select.py [options] source_file:/table_name dest_file:/table_name

Options:
  -h  print this message.
  -e  selection expression.
  -f  fields.
  -c  compression level (0 - 9).
  -l  compression library (default: zlib).
  -p  enable profiling and save result.
  -b  chunksize in bytes, suffix as 'k', 'm' and 'g' are supported.

"""
import tables
import sys
import numpy as np
from numpy.lib.recfunctions import repack_fields
from time import time
from os import path
from getopt import gnu_getopt
from multiprocessing import cpu_count

def select_table(source, dest, selection, fields=None, complevel=0, complib='zlib', chunksize=None, profiling=None):
    file_in, node_in = source.split(':')
    file_out, node_out = dest.split(':')
    h5_in = tables.open_file(file_in, 'r', max_blosc_threads=(1+2*cpu_count()))
    tab_in = h5_in.get_node(node_in)
    if fields is None:
        tab_out_dtype = tab_in.dtype
    else:
        a = np.empty((1,), dtype=tab_in.dtype)
        tab_out_dtype = repack_fields(a[fields]).dtype
    if profiling is not None:
        iops = open(profiling, 'w')
        iops.write(u'bytes,rows,timestamp\n')
    with tables.open_file(file_out, 'a', max_blosc_threads=(1+2*cpu_count())) as h5_out:
        if complevel == 0:
            filters = None
        else:
            filters = tables.Filters(complevel=complevel, complib=complib)
        if chunksize is None:
            chunkshape = None
        else:
            chunkshape = (max(1, chunksize//tab_in.rowsize), )
        grpname, tabname = path.split(node_out)
        tab_out = h5_out.create_table(
            grpname,
            tabname,
            tab_out_dtype,
            title         = tab_in.title,
            filters       = filters,
            expectedrows  = tab_in.nrows,
            createparents = True,
            chunkshape    = chunkshape
        )
        nb = max(tab_in.chunkshape[0], tab_out.chunkshape[0])
        t = 0
        tic = time()
        while t<tab_in.nrows:
            n = min(tab_in.nrows-t, nb)
            if selection is None:
                a = tab_in.read(start=t, stop=t+n)
            else:
                a = tab_in.read_where(selection, start=t, stop=t+n)
            if fields is None:
                tab_out.append(a)
            else:
                tab_out.append(repack_fields(a[fields]))
            t += n
            sys.stdout.write(u'\rSaving selected table {:d}/{:d} rows ({:.1f}%, {:.2f} MRows/s, {:.2f} GiB/s)......'.format(t, tab_in.nrows, 100.0*t/tab_in.nrows, 1e-6*t/(time()-tic), 1e-9*t*tab_in.rowsize/(time()-tic)))
            sys.stdout.flush()
            if profiling is not None:
                iops.write(u'{:d},{:d},{:f}\n'.format(int(n*tab_in.rowsize), int(n), time()-tic))
        sys.stdout.write(u'\rSaving selected table {:d}/{:d} rows ({:.1f}%, {:.2f} MRows/s, {:.2f} GiB/s)......OK\n'.format(t, tab_in.nrows, 100.0*t/tab_in.nrows, 1e-6*t/(time()-tic), 1e-9*t*tab_in.rowsize/(time()-tic)))
        sys.stdout.flush()
        if profiling is not None:
            iops.close()
    print(u'Selected table saved to {}:{}.'.format(file_out, node_out))

if __name__ == '__main__':
    opts, args = gnu_getopt(sys.argv[1:], 'he:c:l:b:p:f:')
    complevel = 0
    complib = 'zlib'
    chunksize = None
    selection = None
    profiling = None
    fields    = None
    for opt, val in opts:
        if opt == '-h':
            print(__doc__)
            sys.exit()
        elif opt == '-e':
            selection = val
        elif opt == '-c':
            complevel = int(val)
        elif opt == '-l':
            complib = val
        elif opt == '-p':
            profiling = val
        elif opt == '-f':
            fields = val.split(',')
        elif opt == '-b':
            if val.lower().endswith('k'):
                chunksize = int(int(val[:-1]) * 1024)
            elif val.lower().endswith('m'):
                chunksize = int(int(val[:-1]) * 1024**2)
            elif val.lower().endswith('g'):
                chunksize = int(int(val[:-1]) * 1024**3)
            else:
                chunksize = int(val)
    source = args[0]
    dest   = args[1]
    select_table(source, dest, selection, fields=fields, complevel=complevel, complib=complib, chunksize=chunksize, profiling=profiling)
