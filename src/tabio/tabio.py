#!/usr/bin/env python
"""Convert table in the input file to specified format.
Syntax:
tabio.py input_file:input_table output_file[:output_table] [option=value]

Options:
--format[-f]=FORMAT Format of output file. Supported formats:
                    ROOT[Tree, TTree]: Table implemented with ROOT TFile contains TTree object.
                    HDF5: Table implemented with HDF5 datasets contained in the same group.
                    TABLES[TABLE, PyTables]: Native PyTables format.

--start=START       From START row to STOP row, by STEP of rows.
--stop=STOP
--step=STEP
-s=START:STOP:STEP

--sample[-r]=SAMPLE_RATE Randomly sample input entries.

--mode[-m]=MODE     Output mode. Supported modes:
                    'Create'['w', 'new']
                    'Update'['a', 'append']

"""
import sys
import h5py
import tables
import numpy as np
from six import iteritems
from multiprocessing import cpu_count
from ROOT import TFile,TTree,TDirectoryFile
from os import path
from array import array
from time import time
from root_numpy import array2tree,tree2array

tables.set_blosc_max_threads(cpu_count())

default_buffer_size_bytes = 32*1024**2
numpy_type_to_root_type = {
    'string' :'C',
    'int8'   :'B',
    'uint8'  :'b',
    'int16'  :'S',
    'uint16' :'s',
    'int32'  :'I',
    'uint32' :'i',
    'float32':'F',
    'float64':'D',
    'int64'  :'L',
    'uint64' :'l',
    'bool'   :'O'
    }
numpy_type_to_python_type = {
    'int8'   :'b',
    'uint8'  :'B',
    'int16'  :'h',
    'uint16' :'H',
    'int32'  :'i',
    'uint32' :'I',
    'float32':'f',
    'float64':'d',
    'int64'  :'l',
    'uint64' :'L',
    'bool'   :'b'
    }
root_type_to_numpy_type = {
    'C':'string',
    'B':'int8',
    'b':'uint8',
    'S':'int16',
    's':'uint16',
    'I':'int32',
    'i':'uint32',
    'F':'float32',
    'D':'float64',
    'L':'int64',
    'l':'uint64',
    'O':'bool'
    }
numpy_type_to_root_type = {
    'string' :'C',
    'int8'   :'B',
    'uint8'  :'b',
    'int16'  :'S',
    'uint16' :'s',
    'int32'  :'I',
    'uint32' :'i',
    'float32':'F',
    'float64':'D',
    'int64'  :'L',
    'uint64' :'l',
    'bool'   :'O'
    }
root_type_name_to_python_type={
    'Bool_t'   :'b',
    'Char_t'   :'b',
    'UChar_t'  :'B',
    'Short_t'  :'h',
    'UShort_t' :'H',
    'Int_t'    :'i',
    'UInt_t'   :'I',
    'Float_t'  :'f',
    'Double_t' :'d',
    'Long64_t' :'l',
    'ULong64_t':'L'
    }
root_type_name_to_numpy_type={
    'Bool_t'   :'bool',
    'Char_t'   :'int8',
    'UChar_t'  :'uint8',
    'Short_t'  :'int16',
    'UShort_t' :'uint16',
    'Int_t'    :'int32',
    'UInt_t'   :'uint32',
    'Float_t'  :'float32',
    'Double_t' :'float64',
    'Long64_t' :'int64',
    'ULong64_t':'uint64'
    }
root_file_mode={
    'create'  :'create',
    'new'     :'create',
    'update'  :'update',
    'append'  :'update',
    'a'       :'update',
    'recreate':'recreate',
    'write'   :'recreate',
    'w'       :'recreate',
    'read'    :'read',
    'readonly':'read'
    }
pytables_file_mode={
    'update'  :'a',
    'a'       :'a',
    'recreate':'w',
    'write'   :'w',
    'create'  :'w',
    'w'       :'w',
    'read'    :'r',
    'readonly':'r'
    }
hdf5_file_mode={
    'update'  :'a',
    'a'       :'a',
    'recreate':'w',
    'create'  :'w',
    'w-'      :'w',
    'write'   :'w',
    'w'       :'w',
    'r'       :'r',
    'read'    :'r',
    'readonly':'r'
    }
def create_groups(h5file, group):
    parent_obj = h5file.get_node('/')
    groups = group.split('/')
    while groups:
        g = groups.pop(0)
        if g not in ['','/']:
            try:
                parent_obj = h5file.get_node(parent_obj, g)
            except:
                parent_obj = h5file.create_group(parent_obj, g)
    return parent_obj

def convert_table(input_fname,input_tname,output_fname=None,mode='create',output_format=None,output_tname=None,start=None,stop=None,step=None,samplerate=1.0):
    """Convert input table from input format to specified output format.
    """

    #
    # parse input
    _,extname = path.splitext(input_fname)
    if extname.lower() == '.root':
        tabin = tree_table(fname=input_fname,tname=input_tname,mode='readonly')
    elif tables.is_hdf5_file(input_fname):
        if tables.is_pytables_file(input_fname):
            ifile = tables.open_file(input_fname,'r')
            tabin = ifile.get_node(path.join('/',input_tname))
        else:
            tabin = hdf5_table(fname=input_fname,tname=input_tname,mode='r')
    else:
        raise TypeError('Unrecognized file format: %s.'%input_fname)

    nrows_in = tabin.nrows
    dtype    = tabin.dtype
    print('Input table contains %d rows.'%nrows_in)

    if not start:
        start = 0
    if not stop:
        stop  = nrows_in
    if not step:
        step  = 1
    nrows_out = int(np.ceil(1.0*(stop-start)/step))

    #
    # parse output
    if output_fname:
        _,extname = path.splitext(output_fname)
        if extname.lower() == '.root':
            output_format = 'root'
    if output_format:
        if not output_fname:
            if output_format.lower() in ['h5','hdf5','table','tables','pytables']:
                output_fname = path.splitext(input_fname)[0] + '.h5'
            if output_format.lower() in ['root','tree','ttree']:
                output_fname = path.splitext(input_fname)[0] + '.root'
    if not output_tname:
        output_tname = input_tname


    if output_format.lower() in ['h5','hdf5']:
        tabout = hdf5_table(fname=output_fname,tname=output_tname,mode=hdf5_file_mode[mode],row_dtype=dtype,nrows_max=nrows_out)
    elif output_format.lower() in ['root','tree','ttree']:
        tabout = tree_table(fname=output_fname,tname=output_tname,mode=root_file_mode[mode],row_dtype=dtype)
    elif output_format.lower() in ['table','tables','pytables']:
        ofile  = tables.open_file(output_fname,pytables_file_mode[mode])
        tdir,tname = path.split(output_tname)
        parent_obj = create_groups(ofile, tdir)
        tabout = ofile.create_table(parent_obj,tname,
            description=tables.descr_from_dtype(dtype)[0],
            expectedrows=int(nrows_out*samplerate/step),
            filters=tables.Filters(complevel=5,complib='blosc'))
    else:
        raise TypeError('Unsupported output format %s.'%output_format)

    #
    # transfer data
    t    = start
    tic  = time()
    nbuf = default_buffer_size_bytes / tabin.rowsize
    while t < stop:
        n    = int(min(nbuf, int(stop - t)))
        rows = tabin.read(t,t+n,step)
        if samplerate<1.0:
            accepted    = np.bool_(np.random.rand(rows.size)<samplerate)
            nsamples    = np.sum(accepted)
            rows_sample = np.empty(nsamples, dtype=rows.dtype)
            for field in rows.dtype.fields:
                rows_sample[field][:] = rows[field][accepted]
            rows = rows_sample
        tabout.append(rows)
        t   += n
        sys.stdout.write('\r%d (%.2f%%) rows processed. %.2f seconds elapsed.'%(t-start,100.0*(t-start)/(stop-start),time()-tic))
        sys.stdout.flush()

    if output_format.lower() in ['root','tree','ttree']:
        #tabout.tree.Fill()
        tabout.file.Write()
        tabout.file.Close()
    print("\nOutput: %s:%s"%(output_fname,output_tname))


class branch_array(object):
    def __init__(self,branch=None,tree=None,fname=None,tname=None,bname=None):
        if not branch:
            if not tree:
                tfile    = TFile(fname,'readonly')
                tree     = tfile.Get(tname)
            branch   = tree.GetBranch(bname)
        tree  = branch.GetTree()
        bname = branch.GetName()
        self.tree    = tree
        self.branch  = branch
        tdirectory   = tree.GetDirectory()
        self.file    = tdirectory.GetFile()
        self.type    = branch.GetLeaf(bname).GetTypeName()
        self.dtype   = np.dtype(root_type_name_to_numpy_type[self.type])
        self.size    = branch.GetEntries()
        self.shape   = (self.size,)
        self.__ptr__ = array(root_type_name_to_python_type[self.type],[0])
        self.branch.SetAddress(self.__ptr__)
    def __len__(self):
        return self.branch.GetEntries()
    def __getitem__(self,i):
        self.branch.SetAddress(self.__ptr__)
        if isinstance(i,int):
            if self.branch.GetEntry(i):
                return self.__ptr__[0]
            else:
                raise IndexError("Index %d out of range."%i)
        elif isinstance(i,slice):
            start,stop,step = i.indices(self.branch.GetEntries())
            n = int(np.ceil((stop - start)/step))
            a = np.empty(n,dtype=root_type_name_to_numpy_type[self.type])
            for k in range(n):
                self.branch.GetEntry(start+k*step)
                a[k] = self.__ptr__[0]
            return a

class sparse_array(object):
    def __init__(self,array,start=None,stop=None,step=None):
        start,stop,step = slice(start,stop,step).indices(len(array))
        self.array = array
        self.start = start
        self.stop  = stop
        self.step  = step
    def __inner_index__(self,i):
        if isinstance(i,int):
            return i*self.step+self.start
        elif isinstance(i,slice):
            istart = max(i.start,0)
            istep  = max(i.step, 1)
            if not i.stop:
                istop = self.size
            start = istart*self.step + self.start
            stop  = istop *self.step + self.start
            step  = istep *self.step
            return slice(start,stop,step)
    def __getitem__(self,i):
        return self.array.__getitem__(self.__inner_index__(i))
    def __setitem__(self,i,v):
        return self.array.__setitem__(self.__inner_index__(i),v)
    def __len__(self):
        return int(np.ceil(1.0*(self.stop-self.start)/self.step))

class table(object):
    def __init__(self,columns):
        self.cols    = {}
        self.dtype   = []
        self.rowsize = 0
        self.nrows   = np.inf
        for key,val in iteritems(columns):
            self.cols[key] = val
            self.dtype.append((key.encode('ascii'),np.dtype(val.dtype)))
            self.rowsize += np.dtype(val.dtype).itemsize
            self.nrows = int(min(self.nrows, val.size))
        self.dtype = np.dtype(self.dtype)
    def __getitem__(self,key):
        return self.cols[key]
    def __add__(self,tab):
        cols = {}
        for key,val in iteritems(self.cols):
            cols[key] = val
        for key,val in iteritems(tab.cols):
            if not cols.has_key(key):
                cols[key] = val
        return table(cols)
    def read(self,start=None,stop=None,step=None):
        start = max(start,0)
        step  = max(step, 1)
        if not stop:
            stop = self.nrows
        n = int(np.ceil(1.0*(stop-start)/step))
        arr = np.empty(n,dtype=self.dtype)
        for key,val in iteritems(self.cols):
            arr[key] = val[start:stop:step]
        return arr

class hdf5_table(table):
    def __init__(self,fname=None,tname=None,mode="r",nrows_max=None,row_dtype=None,chunks=True,compression="lzf"):
        mode = hdf5_file_mode[mode]
        self.cols     = {}
        self.dtype    = []
        self.rowsize  = 0
        self.writable = False
        if mode.lower() in ['r', 'read', 'readonly']:
            self.nrows = np.inf
            self.file = h5py.File(fname,'r')
            self.group = self.file.require_group(tname)
            if row_dtype:
                for cname,ctype in iteritems(row_dtype.fields):
                    ctype = ctype[0]
                    self.cols[cname] = self.group.require_dataset(cname,dtype=ctype)
                    self.nrows = int(min(self.nrows, self.cols[cname].size))
                    self.nrows_max = self.nrows
                    self.dtype.append((cname.encode('ascii'), np.dtype(self.cols[cname].dtype)))
                    self.rowsize += np.dtype(self.cols[cname].dtype).itemsize
            else:
                for cname,col in iteritems(self.group):
                    if len(col.shape) == 1:
                        self.cols[cname] = col
                        self.nrows = int(min(self.nrows, col.size))
                        self.nrows_max = self.nrows
                        self.dtype.append((cname.encode('ascii'), np.dtype(col.dtype)))
                        self.rowsize += np.dtype(col.dtype).itemsize
        elif mode.lower() in ['a', 'append', 'update']:
            self.writable = True
            self.file = h5py.File(fname,'a')
            try:
                self.group = self.file.create_group(tname)
                self.nrows = 0
            except ValueError:
                self.group = self.file.require_group(tname)
                try:
                    self.nrows = int(self.group.attrs['nrows'])
                except:
                    self.nrows = np.inf
            if row_dtype:
                for cname,ctype in iteritems(row_dtype.fields):
                    ctype = ctype[0]
                    self.cols[cname] = self.group.require_dataset(cname,shape=(nrows_max,),dtype=ctype,chunks=chunks,compression=compression)
                    self.nrows = int(min(self.nrows, self.cols[cname].size))
                    self.nrows_max = int(min(nrows_max, self.cols[cname].size))
                    self.dtype.append((cname.encode('ascii'), np.dtype(self.cols[cname].dtype)))
                    self.rowsize += np.dtype(self.cols[cname].dtype).itemsize
            else:
                self.nrows_max = nrows_max
                if not self.nrows_max:
                    self.nrows_max = np.inf
                for cname,col in iteritems(self.group):
                    if len(col.shape) == 1:
                        self.cols[cname] = col
                        self.nrows = int(min(self.nrows, col.size))
                        self.nrows_max = int(min(self.nrows_max, col.size))
                        self.dtype.append((cname.encode('ascii'), np.dtype(col.dtype)))
                        self.rowsize += np.dtype(col.dtype).itemsize
        elif mode.lower() in ['w', 'write', 'recreate']:
            self.writable = True
            self.nrows     = 0
            self.nrows_max = int(nrows_max)
            self.file = h5py.File(fname,'w')
            self.group = self.file.require_group(tname)
            for cname,ctype in iteritems(row_dtype.fields):
                ctype = ctype[0]
                self.cols[cname] = self.group.create_dataset(cname,shape=(self.nrows_max,),dtype=ctype,chunks=chunks,compression=compression)
                self.dtype.append((cname.encode('ascii'), np.dtype(self.cols[cname].dtype)))
                self.rowsize += np.dtype(self.cols[cname].dtype).itemsize
        else:
            raise StandardError('unrecognized mode %s'%mode)
        self.dtype = np.dtype(self.dtype)

    def append(self,rows):
        if self.writable and (self.nrows < self.nrows_max):
            n = rows.size
            t = self.nrows
            for key in rows.dtype.fields:
                self.cols[key][t:t+n] = rows[key][:]
            self.nrows += n
            self.group.attrs['nrows'] = self.nrows
        else:
            raise StandardError("Table is read-only or out of space.")

    def read(self,start=None,stop=None,step=None):
        start = max(start,0)
        step  = max(step, 1)
        if not stop:
            stop = self.nrows
        stop  = min(stop, self.nrows)
        n = int(np.ceil(1.0*(stop-start)/step))
        arr = np.empty(n,dtype=self.dtype)
        for key,val in iteritems(self.cols):
            arr[key] = val[start:stop:step]
        return arr

class tree_table(table):
    def __init__(self,tree=None,fname=None,tname=None,mode="read",row_dtype=None):
        mode = root_file_mode[mode]
        self.open_file = False
        if not tree:
            if path.exists(fname):
                tfile = TFile(fname,mode)
                tree  = tfile.Get(tname)
            else:
                tfile = TFile(fname,'create')
            if not isinstance(tree,TTree): # tree doesn't exist. create it.
                tdir,tname = path.split(tname)
                tree = TTree(tname, '')
                for bname, btype in iteritems(np.dtype(row_dtype).fields):
                    btype = btype[0].name
                    addr  = array(numpy_type_to_python_type[btype],[0])
                    tree.Branch(bname,addr,'%s/%s'%(bname,numpy_type_to_root_type[btype]))
                if tdir:
                    parent_obj = tfile
                    dirs = tdir.split('/')
                    while dirs:
                        d = dirs.pop(0)
                        if d not in ['','/']:
                            parent_obj = TDirectoryFile(d,'','',parent_obj)
                    tree.SetDirectory(tfile.GetDirectory(tdir))
                else:
                    tree.SetDirectory(tfile)
            self.open_file = True
        self.tree  = tree
        self.file  = tfile
        self.cols  = {}
        self.dtype = []
        self.rowsize = 0
        self.nrows   = np.inf
        for branch in tree.GetListOfBranches():
            bname  = branch.GetName()
            barray = branch_array(branch=branch)
            self.cols[bname] = barray
            self.dtype.append((bname,np.dtype(barray.dtype)))
            self.rowsize += np.dtype(barray.dtype).itemsize
            self.nrows = int(min(self.nrows, barray.size))
        self.dtype=np.dtype(self.dtype)

    def read(self,start=None,stop=None,step=None,cols=None,condition=None):
        return tree2array(self.tree, branches=cols, selection=condition, start=start, stop=stop, step=step)

    def append(self,rows):
        self.tree = array2tree(rows, tree=self.tree)

def print_table(t,title):
    print("{:-^80}".format(' '+title+' '))
    print(" {:<5} | {:<15} | {:<60}".format('Index','Type','Name'))
    print("{:-^80}".format(""))
    i = 0
    for key,val in iteritems(t.read(0,1).dtype.fields):
        val = val[0].name
        try:
            print(" {:<5} | {:<15} | {:<60}".format(i, val, key))
            i += 1
        except:
            pass
    print("{:-^80}".format(""))

if __name__ == '__main__': #executed from command line
    try:
        args    = []
        options = {}
        for arg in sys.argv[1:]:
            if '--format=' in arg:
                options['output_format'] = arg.split('=')[1]
            elif '-f=' in arg:
                options['output_format'] = arg.split('=')[1]
            elif '-s=' in arg:
                start,stop,step = arg.split('=')[1].split(':')
                options['start'] = start
                options['stop']  = stop
                options['step']  = step
            elif '--start=' in arg:
                options['start'] = arg.split('=')[1]
            elif '--stop=' in arg:
                options['stop'] = arg.split('=')[1]
            elif '--step=' in arg:
                options['step'] = arg.split('=')[1]
            elif '--mode=' in arg:
                options['mode'] = arg.split('=')[1]
            elif '-m=' in arg:
                options['mode'] = arg.split('=')[1]
            elif '--sample=' in arg:
                options['samplerate'] = float(arg.split('=')[1])
            elif '-r=' in arg:
                options['samplerate'] = float(arg.split('=')[1])
            else:
                args.append(arg)
        try:
            input_fname, input_tname   = args[0].split(':')
        except:
            input_fname = args[0]
            input_tname = '/'
        try:
            output_fname, output_tname = args[1].split(':')
            options['output_fname'] = output_fname
            options['output_tname'] = output_tname
        except:
            options['output_fname'] = args[1]
        convert_table(input_fname, input_tname, **options)
    except IndexError:
        print(__doc__)
