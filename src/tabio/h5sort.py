#!/usr/bin/env python3
#coding=utf-8
"""Sort and save HDF5 table to specified container.

Syntax:
  h5sort.py [options] source_file:/table_name dest_file:/table_name

Options:
  -h  print this message.
  -s  sortby.
  -r  sort in reversed order (descending order).
  -i  force index sortby column if it is not indexed.
  -c  compression level (0 - 9).
  -l  compression library (default: zlib).
  -p  enable profiling and save result to a csv file.
  -b  chunksize in bytes, suffix as 'k', 'm' and 'g' are supported.

"""
import tables
import sys
from os import path
from getopt import gnu_getopt
from multiprocessing import cpu_count
from time import time

def sort_table(source, dest, sortby, index=True, descorder=False, complevel=0, complib='zlib', chunksize=None, profiling=None):
    file_in, node_in = source.split(':')
    file_out, node_out = dest.split(':')
    h5_in = tables.open_file(file_in, 'r')
    tab_in = h5_in.get_node(node_in)
    if not tab_in.cols.__getattribute__(sortby).is_indexed:
        print(u'{} is not indexed.'.format(sortby))
        if not index:
            print(u'Goodbye.')
            sys.exit()
        else:
            sys.stdout.write(u'Creating completely sorted index (CSI) for {}......'.format(sortby))
            sys.stdout.flush()
            h5_in.close()
            h5_in = tables.open_file(file_in, 'a')
            tab_in = h5_in.get_node(node_in)
            tab_in.cols.__getattribute__(sortby).create_csindex()
            h5_in.close()
            sys.stdout.write(u'\rCreating completely sorted index (CSI) for {}......OK\n'.format(sortby))
            sys.stdout.flush()
            h5_in = tables.open_file(file_in, 'r')
            tab_in = h5_in.get_node(node_in)
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
            tab_in.dtype,
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
            if descorder:
                a = tab_in.read_sorted(sortby, start=int(tab_in.nrows-1-t-n), stop=int(tab_in.nrows-1-t))[::-1]
            else:
                a = tab_in.read_sorted(sortby, start=t, stop=t+n)
            tab_out.append(a)
            t += n
            sys.stdout.write(u'\rSaving sorted table {:d}/{:d} rows ({:.1f}%, {:.2f} MRows/s, {:.2f} GiB/s)......'.format(t, tab_in.nrows, 100.0*t/tab_in.nrows, 1e-6*t/(time()-tic), 1e-9*tab_in.rowsize*t/(time()-tic)))
            sys.stdout.flush()
            if profiling is not None:
                iops.write(u'{:d},{:d},{:f}\n'.format(int(t*tab_in.rowsize), int(t), time()-tic))
        sys.stdout.write(u'\rSaving sorted table {:d}/{:d} rows ({:.1f}%, {:.2f} MRows/s, {:.2f} GiB/s)......OK\n'.format(t, tab_in.nrows, 100.0*t/tab_in.nrows, 1e-6*t/(time()-tic), 1e-9*tab_in.rowsize*t/(time()-tic)))
        sys.stdout.flush()
        if profiling is not None:
            iops.close()
    print(u'Sorted table saved to {}:{}.'.format(file_out, node_out))

if __name__ == '__main__':
    opts, args = gnu_getopt(sys.argv[1:], 'hs:irc:l:b:p:')
    index  = False
    complevel = 0
    complib = 'zlib'
    chunksize = None
    profiling = None
    descorder = False
    for opt, val in opts:
        if opt == '-h':
            print(__doc__)
            sys.exit()
        elif opt == '-s':
            sortby = val
        elif opt == '-i':
            index = True
        elif opt == '-r':
            descorder = True
        elif opt == '-c':
            complevel = int(val)
        elif opt == '-l':
            complib = val
        elif opt == '-p':
            profiling = val
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
    sort_table(source, dest, sortby, index=index, complevel=complevel, complib=complib, chunksize=chunksize, profiling=profiling)
