"""
SD4

This module contains only one function, SD4, which does joint diagonalization of cumulant matrices of order 4 to decorrelate the signals in spatial domain. It allows us to extract signals which are as independent as possible and which was not obtained while performing SD2.
"""

import numpy as np
from sys import stdout 
from numpy import abs, append, arange, arctan2, argsort, array, concatenate, \
                  cos, diag, dot, eye, float32, float64, matrix, multiply, ndarray, newaxis, \
                  sign, sin, sqrt, zeros, ones
from numpy.linalg import eig, pinv
import pyemma.msm as msm
import pyemma.coordinates as coor 
import numpy
import matplotlib.pyplot as plt
import warnings

def SD4(Y,m=None, U=None, verbose=True):
 
    """
    SD4 - Spatial Decorrelation of 4th order of real signals 
	    
    Parameters:
    
        Y -- an m x T spatially whitened matrix (m subspaces, T samples). May be a numpy 
                array or matrix where 
                m: number of subspaces we are interested in.
                T: Number of snapshots of MD trajectory
    
        m -- It is the number of subspaces we are interested in. Defaults to None, in
        which case m=n.
        
        U -- whitening matrix obtained after doing the PCA analysis on m components
              of real data
        
        verbose -- print info on progress. Default is True.
    
    Returns:
    
        W -- separating matrix which is spatially decorrelated of 4th order
        
    
    """
    
    # GB: we do some checking of the input arguments and copy data to new
    # variables to avoid messing with the original input. We also require double
    # precision (float64) and a numpy matrix type for X.
    
    warnings.simplefilter("ignore", np.ComplexWarning)
    [n,T] = Y.shape # GB: n is number of input signals, T is number of samples
    
    if m==None:
        m=n 	# Number of sources defaults to # of sensors
    assert m<=n,\
        "SD4 -> Do not ask more sources (%d) than sensors (%d )here!!!" % (m,n)
        
    print "4th order Spatial Decorrelation -> Estimating cumulant matrices"
    
    # Reshaping of the data, hoping to speed up things a little bit...
    
    Y = Y.T
    
    dimsymm = ((m) * (( m )+ 1)) / 2	# Dim. of the space of real symm matrices
    nbcm = dimsymm  # number of cumulant matrices
    CM = matrix(zeros([(m),(m)*nbcm], dtype=float64)) # Storage for cumulant matrices
    R = matrix(eye((m), dtype=float64))
    Qij = matrix(zeros([m,m], dtype=float64)) # Temp for a cum. matrix
    Xim	= zeros(m , dtype=float64) # Temp
    Xijm = zeros(m , dtype=float64) # Temp
    #Uns = numpy.ones([1,m], dtype=numpy.uint32)    # for convenience
    # GB: we don't translate that one because NumPy doesn't need Tony's rule
    
    # I am using a symmetry trick to save storage.  I should write a short note one of these
    # days explaining what is going on here.
    Range = arange(m) # will index the columns of CM where to store the cum. mats.
    
    # Removing 4th order spatial decorrelations 
    for im in range(m):
        
        Xim = Y[:,im]
        Xijm = multiply(Xim, Xim)
             
        # Note to myself: the -R on next line can be removed: it does not affect
        # the joint diagonalization criterion
        
        Qij = dot(multiply(Xijm , Y).T  , Y) / float(T)  - R - 2 * dot(R[:,im], R[:,im].T)
        
        # To ensure symmetricity of the covariance matrix a mathematical  computation is done
        Qij = (Qij + Qij.T)/2
        CM[:,Range]	= Qij    
        Range = Range  + m 
        for jm in range(im):
                Xijm = multiply(Xim , Y[:, jm ])
                Qij = sqrt(2) * dot(multiply(Xijm, Y).T , Y) / float(T) \
                    - R[:,im] * R[:,jm].T - R[:,jm] * R[:,im].T
                
                # To ensure symmetricity of the covariance matrix a mathematical  computation is done
                Qij = (Qij + Qij.T)/2
                CM[:,Range]	= Qij
                Range = Range + m 
    
    
    nbcm = int(nbcm)
    # Now we have nbcm = m(m+1)/2 cumulants matrices stored in a big m x m*nbcm array.
    V = matrix(eye(m, dtype=float64))
        
    Diag = zeros(m, dtype=float64)
    On = 0.0
    Range = arange(m)
    for im in range(nbcm):
        Diag = diag(CM[:,Range])
        On = On + (Diag*Diag).sum(axis=0)
        Range = Range + m
    Off = (multiply(CM,CM).sum(axis=0)).sum(axis=0) - On
    
    seuil = 1.0e-6 / (sqrt(T)) # % A statistically scaled threshold on `small" angles
    encore = True
    sweep = 0 # % sweep number
    updates = 0 # % Total number of rotations
    upds = 0 # % Number of rotations in a given s
    g = zeros([2,nbcm], dtype=float64)
    gg = zeros([2,2], dtype=float64)
    G = zeros([2,2], dtype=float64)
    c = 0
    s = 0
    ton	= 0
    toff = 0
    theta = 0
    Gain = 0
    
    # Joint diagonalization proper
    
    if verbose:
        print >> stdout, "TD4 -> Contrast optimization by joint diagonalization"
    
    while encore:
        encore = False
        if verbose:
            print >> stdout, "TD4 -> Sweep #%3d" % sweep ,
        sweep = sweep + 1
        upds  = 0
        Vkeep = V
      
        for p in range(m-1):
            for q in range(p+1, m):
                
                Ip = arange(p, m*nbcm, m)
                Iq = arange(q, m*nbcm, m)
                
                # computation of Givens angle
                g = concatenate([CM[p,Ip] - CM[q,Iq], CM[p,Iq] + CM[q,Ip]])
                gg = dot(g, g.T)
                ton = gg[0,0] - gg[1,1] 
                toff = gg[0,1] + gg[1,0]
                theta = 0.5 * arctan2(toff, ton + sqrt(ton * ton + toff * toff))
                Gain = (sqrt(ton * ton + toff * toff) - ton) / 4.0
                
                # Givens update
                if abs(theta) > seuil:
                    encore = True
                    upds = upds + 1
                    c = cos(theta) 
                    s = sin(theta)
                    G = matrix([[c, -s] , [s, c] ])
                    pair = array([p,q])
                    V[:,pair] = V[:,pair] * G
                    CM[pair,:] = G.T * CM[pair,:]
                    CM[:,concatenate([Ip,Iq])] = \
                        append( c*CM[:,Ip]+s*CM[:,Iq], -s*CM[:,Ip]+c*CM[:,Iq], \
                               axis=1)
                    On = On + Gain
                    Off = Off - Gain
                    
        if verbose:
            print >> stdout, "completed in %d rotations" % upds
        updates = updates + upds
    if verbose:
        print >> stdout, "TD4 -> Total of %d Givens rotations" % updates
    
    # A separating matrix
    # ===================
    
    W = V.T * U
    
    # Permute the rows of the separating matrix B to get the most energetic components first.
    # Here the **signals** are normalized to unit variance.  Therefore, the sort is
    # according to the norm of the columns of A = pinv(W)

    if verbose:
        print >> stdout, "TD4 -> Sorting the components"
    
    A = pinv(W)
    keys =  array(argsort(multiply(A,A).sum(axis=0)[0]))[0]
    W = W[keys,:]
    W = W[::-1,:]     # % Is this smart ?
    
    
    if verbose:
        print >> stdout, "TD4 -> Fixing the signs"
    b	= W[:,0]
    signs = array(sign(sign(b)+0.1).T)[0] # just a trick to deal with sign=0
    W = diag(signs) * W
    print W.shape
    return W

    """
      
      Notes:
      ** Here we consider signals which are spatially decorrelated of 
      order 2
      
      """
