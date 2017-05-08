"""
SD2

This module contains only one function, SD2, which performs spatial decorrelation of 2nd order.
"""

import numpy
from sys import stdout 
from numpy import argsort, cov, diag, dot, sqrt, absolute, arange
from numpy.linalg import inv, eig, eigh
from numpy import sqrt, ndarray, matrix, float64

#perform 2nd order spatial decorrelation
def SD2(data,m= None,verbose=True):
    
    """
    SD2 - Spatial Decorrelation of 2nd order of real signals 
    
    Parameters:
    
        data -- a 3n x T data matrix (number 3 is due to the x,y,z coordinates for each atom). May be a numpy array or
                matrix where 
				
                n: size of the protein
                T: Number of snapshots of MD trajectory
    
        m -- dimensionality of the subspace we are interested in. Default value is None, in
        which case m=n.
        
        verbose -- print information on progress. Default is True.
    
    Returns:
    
        A 3n x m matrix U (NumPy matrix type), such that Y = U x data is a 2nd order
        spatially whitened source extracted from the 3n x T data matrix 'data'. If m is
        omitted, U is a square 3n x 3n matrix (as many sources as sensors). 
         
        Ds: has eigen values sorted by increasing variance
		PCs: holds the index for m most significant principal components by decreasing variance
        S = Ds[PCs] 
		
		S – Eigen values of the ‘data’ covariance matrix 
        B - Eigen vectors of the 'data' covariance matrix. The eigen vectors are orthogonal.
        """
    
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
    
    # remove the mean from data
    data -= data.mean(1)
    
    # whitening & projection onto signal subspace
    # ===========================================
    if verbose:
        print >> stdout, "2nd order Spatial Decorrelation -> Whitening the data"
    [D,U] = eigh((data * data.T) / float(T)) # An eigen basis for the sample covariance matrix
    k = D.argsort()
    Ds = D[k] # Sort by increasing variances
    PCs = arange(n-1, n-m-1, -1)    # The m most significant princip. comp. by decreasing variance
    S = Ds[PCs]
    # --- PCA  ----------------------------------------------------------
    B = U[:,k[PCs]].T    # % At this stage, B does the PCA on m components
 
    # --- Scaling  ------------------------------------------------------
    scales = sqrt(S) # The scales of the principal components .
    U = diag(1./scales) * B  # Now, B does PCA followed by a rescaling = sphering
    # --- Sphering ------------------------------------------------------

    Y = U * data # %% We have done the easy part: B1 is a whitening matrix and Y is white.
    return (Y, S, B.T, U)

    """
    NOTE: At this stage, Y is spatially whitened by performing PCA analysis on m components of the real data
    Y is now a matrix of spatially uncorrelated components.
    """
