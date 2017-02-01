# DNCuts

This code is a python implementation of the matlab code presented [here](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/mcg/) and found [here](https://github.com/jponttuset/mcg/tree/master/dncuts).

### What this is
While most of it is a direct transcription from matlab to python (with some reorganization), I've added some code that will allow you to deal with considerably sized matrices.  The matrix multiplication in the original DNCuts was giving me memory issues because of the large matrices I was working with (and because I only had 4GB RAM and 4GB Swap at the time...), so the code will now perform blockwise sparse matrix multiplication while memory mapping the partial products along the way and killing values below a low threshold (otherwise I was getting non-zero density at least 2 orders of magnitude higher in the product).  The matlab code / paper also detailed a kind of a weird norm that wasn't included in numpy, and I couldn't envision a way to vectorize, so there is a c-extension which does the normalization.

### What the files are
[`main.py`](./main.py) is the code that should be called from the CLI  
[`dncuts.py`](./dncuts.py) houses all the functions dncuts and ncuts call  
[`lena.bmp`](./lena.bmp) is an example image  
`_norm_c.so` is a compiled Shared Object for the normalization  
[`c-extensions`](./c-extensions/) houses all the source files for the normalization code  
[`config.yaml`](./config.yaml) is the configuration file.  I find yaml's to be a very readable configuration format.  

### Running

`$ python main.py` runs it on [`lena.bmp`](./lena.bmp) using the default config.

Example config file:  
```
image: 'lena.bmp'
size:
    - 256
    - 256
numblock: 8
n_eigv: 16
saveDir: 'savefiles/'
figDir: 'savefiles/figures'
logfile: 'log.txt'
```

Alternatively you could set:  
```
aff: 'path/to/my/affinity/sparsecscmatrix.npz'
```
where the file is a saved python dictionary with the keys: `data`,`indices`,`indptr`, and `shape` which correspond to the constituent vectors / shape of your CSC format sparse matrix.

There are also a few flags you can use at runtime.

### Contact
Email or submit an issue for questions.
