import numpy
from sys import stdout 
from numpy import argsort, cov, diag, dot, sqrt, absolute, arange
from numpy.linalg import inv, eig, eigh
from numpy import sqrt, ndarray, matrix, float64
#perform 2nd order spatial decorrelation
def SD2(data,m= None,verbose=True):
    
    # GB: we do some checking of the input arguments and copy data to new
    # variables to avoid messing with the original input. We also require double
    # precision (float64) and a numpy matrix type for data.
    
    assert isinstance(data, ndarray),\
        "data (input data matrix) is of the wrong type (%s)" % type(data)
    origtype = data.dtype # remember to return matrix B of the same type
    data = matrix(data.astype(float64))    
    
    assert data.ndim == 2, "X has %d dimensions, should be 2" % data.ndim
    assert (verbose == True) or (verbose == False), \
        "verbose parameter should be either True or False"
    
    [T,n] = data.shape # GB: n is number of input signals, T is number of samples
    
    if m==None:
        m=n 	# Number of sources defaults to # of sensors
    assert m<=n,\
        "SD2 -> Do not ask more sources (%d) than sensors (%d )here!!!" % (m,n)

    if verbose:
        print >> stdout, "2nd order Spatial Decorrelation -> Looking for %d sources" % m
        print >> stdout, "2nd order Spatial Decorrelation -> Removing the mean value"
    
    data = data.T
    data -= data.mean(1)
    
    # whitening & projection onto signal subspace
    # ===========================================
    if verbose:
        print >> stdout, "2nd order Spatial Decorrelation -> Whitening the data"
    [D,U] = eigh((data * data.T) / float(T)) # An eigen basis for the sample covariance matrix
    k = D.argsort()
    Ds = D[k] # Sort by increasing variances
    PCs = arange(n-1, n-m-1, -1)    # The m most significant princip. comp. by decreasing variance

    # --- PCA  ----------------------------------------------------------
    B = U[:,k[PCs]].T    # % At this stage, B does the PCA on m components

    # --- Scaling  ------------------------------------------------------
    scales = sqrt(Ds[PCs]) # The scales of the principal components .
    B1 = diag(1./scales) * B  # Now, B does PCA followed by a rescaling = sphering
    # --- Sphering ------------------------------------------------------
    
    Y = B1 * data # %% We have done the easy part: B1 is a whitening matrix and Y is white.
    return (Y, Ds[PCs],B.T , B1)