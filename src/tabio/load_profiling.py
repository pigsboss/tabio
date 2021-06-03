#!/usr/bin/env python
#coding=utf-8
"""Load profiling result from h5sort or h5index and make performance plots.

Syntax:
  load_profiling.py [options] csv_file

Options:
  -h  print this message.
  -o  save figures.
  -s  scaling index for b3 spline wavelet smoothing.
  -u  unit: bytes or rows.

Copyright: pigsboss@github
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import csv
from pymath.temporal import dst_b3
from os import path
from getopt import gnu_getopt

def plot_iops(csv_file, output=None, unit='bytes', scaling_index=None):
    iops = {
        'bytes':[],
        'rows':[],
        'timestamp':[]
    }
    with open(csv_file, 'r') as fp:
        reader = csv.DictReader(fp)
        for r in reader:
            for k in r:
                iops[k].append(r[k])
    ts   = np.double(iops['timestamp'])
    if unit == 'bytes':
        perf = dst_b3(1e-9*np.double(iops['bytes'][1:])/np.diff(ts))
    else:
        perf = dst_b3(1e-6*np.double(iops['rows'][1:])/np.diff(ts))
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    if scaling_index is None:
        scaling_index = list(range(len(perf)))
    for i in scaling_index:
        if i == 0:
            label = 'per chunk'
        else:
            label = '{:d} chunks average'.format(int(2**i))
        try:
            ax.plot(ts[:-1], perf[i], label=label)
        except IndexError:
            print(u'maximum scaling index available: {:d}'.format(len(perf)-1))
    ax.set_xlabel('time, in seconds')
    if unit == 'bytes':
        ax.set_ylabel('Throughput, in GiB/s')
    else:
        ax.set_ylabel('Throughput, in MRows/s')
    ax.set_xlim([ts[0], ts[-1]])
    plt.legend()
    plt.tight_layout()
    if output is None:
        plt.show()
    else:
        plt.savefig(output)
if __name__ == '__main__':
    opts, args = gnu_getopt(sys.argv[1:], 'ho:s:u:')
    output = None
    scaling_index = None
    unit = 'bytes'
    for opt, val in opts:
        if opt == '-h':
            print(__doc__)
            sys.exit()
        elif opt == '-o':
            output = val
        elif opt == '-s':
            scaling_index = [int(s) for s in val.split(',')]
        elif opt == '-u':
            unit = val
    csv_file = args[0]
    plot_iops(csv_file, output=output, scaling_index=scaling_index, unit=unit)
