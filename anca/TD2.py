import numpy as np
from sys import stdout
from numpy import argsort, cov, diag, dot, sqrt, std, float64, mean,shape, ndarray, matrix, absolute, arange
from numpy.linalg import inv, eigh, eig
from numpy import sqrt
import warnings


def TD2(Y,m=None, B1 = None, lag=None,verbose=True):
    #perform 2nd order temporal decorrelation
    
    # GB: we do some checking of the input arguments and copy data to new
    # variables to avoid messing with the original input. We also require double
    # precision (float64) and a numpy matrix type for Y.
    warnings.simplefilter("ignore", np.ComplexWarning)
    assert isinstance(Y, ndarray),\
        "Y (input data matrix) is of the wrong type (%s)" % type(Y)
    origtype = Y.dtype # remember to return matrix B of the same type
    Y = matrix(Y.astype(float64))
    assert Y.ndim == 2, "Y has %d dimensions, should be 2" % Y.ndim
    assert (verbose == True) or (verbose == False), \
        "verbose parameter should be either True or False"
    
    [n,T] = Y.shape # GB: n is number of input signals, T is number of samples
    
    if m==None:
        m=n 	# Number of sources defaults to # of sensors
    assert m<=n,\
        "TD2 -> Do not ask more sources (%d) than sensors (%d )here!!!" % (m,n)
    Y = Y.T
   
    Y1 = Y[0: T-lag:]
    Y2 = Y[lag:T:]
    if verbose:
        #   print >> stdout, "2nd order Temporal Decorrelation -> Looking for %d sources" % m
        print >> stdout, "2nd order Temporal Decorrelation -> Removing the mean value"
    
    #compute time-delayed covariance matrix
    Tic = (Y1.T * Y2)/float(T-lag)
    Tic = ((Tic + Tic.T))/2
    
    if verbose:
        print >> stdout, "2nd order Temporal Decorrelation -> Whitening the data"

    [Dtd2,Utd2] = eigh((Tic)) # An eigen basis for the sample covariance matrix
    ktd2 = abs(Dtd2).argsort()
    Dstd2 = Dtd2[ktd2] #sorting by increasing variance
    PCstd2 = arange(n-1, n-m-1, -1)      
    Btd2 = Utd2[:,ktd2[PCstd2]].T    
    
    B2 = dot(Btd2 , B1)
    
    scales = (Dstd2[PCstd2]) # The scales of the principal components .
    B2td2 = dot(diag(1./scales) , Btd2)  
    Z = dot( B2td2 , Y.T ) # %% We have done the easy part: B2 is a whitening matrix and Z is white.
    
    return (Z, Dstd2[PCstd2],Btd2 , B2)