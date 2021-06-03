#!/usr/bin/env python
"""Select and copy rows from source pytables to detination.
Usage:
select_pytables.py input_file:/input_table condition fields (start, stop, step) output_file:/output_table [complevel=0...9]

"""
import tables
import sys
import numpy as np
from os import path
from time import time
from multiprocessing import cpu_count

buffer_size = 256 # buffer size in MBytes
input_file_path, input_object_path = sys.argv[1].split(':')
if len(sys.argv[2]) > 0:
    condition = sys.argv[2]
else:
    condition = None
if len(sys.argv[3]) > 0:
    fields = sys.argv[3].split(',')
else:
    fields = None
if len(sys.argv[4]) > 0:
    input_slice = slice(*eval(sys.argv[3]))
else:
    input_slice = slice(None,None,None)
output_file_path, output_object_path = sys.argv[5].split(':')
output_group, output_object_name = path.split(output_object_path)
try:
    if sys.argv[6].lower()[:10]=='complevel=':
        clvl=int(sys.argv[6][10:])
        filters=tables.Filters(complevel=clvl, complib='blosc')
        ncpu = cpu_count()
        tables.set_blosc_max_threads(ncpu)
except:
    clvl=0
    filters=None

print(u'Buffer size: {:d} MB'.format(buffer_size))
print(u'Source file: {}'.format(input_file_path))
print(u'Source table: {}'.format(input_object_path))
print(u'Destination file: {}'.format(output_file_path))
print(u'Destination table: {}'.format(output_object_path))
print(u'Condition: {}'.format(condition))
if fields is None:
    print(u'All fields included')
else:
    print(u'{} field(s) included only.'.format(fields))
if filters is None:
    print(u'Uncompressed')
else:
    print(u'User specified compression level: {}'.format(clvl))

with tables.open_file(input_file_path, mode='r') as h5in:
    tabin = h5in.get_node(input_object_path)
    if fields is None:
        output_dtype = tabin.dtype
    else:
        output_dtype = np.empty(1,dtype=tabin.dtype)[fields].dtype
    with tables.open_file(output_file_path, mode='w') as h5out:
        tabout = h5out.create_table(
            output_group,
            output_object_name,
            output_dtype,
            title='filtered rows',
            filters=filters,
            createparents=True
        )
        if input_slice.start is None:
            start = 0
        else:
            start = input_slice.start
        if input_slice.stop is None:
            stop = tabin.nrows
        else:
            stop = input_slice.stop
        if input_slice.step is None:
            step = 1
        else:
            step = input_slice.step
        nbrows = int(np.abs(step)*buffer_size*1024**2/tabin.rowsize)
        tic = time()
        t = start
        while t < stop:
            n = min(nbrows, stop-t)
            if condition is None:
                buf = tabin.read(start=t, stop=t+n, step=input_slice.step)
            else:
                buf = tabin.read_where(condition, start=t, stop=t+n, step=input_slice.step)
            if fields is None:
                tabout.append(buf[:])
            else:
                tabout.append(buf[:][fields])
            t += n
            sys.stdout.write(u'  {:12d}/{:d} rows ({:.1f}%) processed, {:.1f} seconds remaining...\r'.format(t,tabin.nrows,100.0*t/tabin.nrows,1.0*(stop-t)*(time()-tic)/t))
            sys.stdout.flush()
print(u'\nFinished.')
