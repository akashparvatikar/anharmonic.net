"""
TD4

This module contains only one function, TD4, which does joint diagonalization of cumulant matrices of order 4 which depends on the time delays. It allows us to extract signals which are as independent as possible and which was not obtained while performing SD2, TD2.
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

def TD4(Z, m=None,V=None, lag=None, verbose=True):
 
    """
    TD4 - Temporal Decorrelation of 4th order of real signals 
    
    Parameters:
    
        Z -- an m x T spatially uncorrelated of order 2 and temporally uncorrelated 
        of order 2 matrix (m subspaces, T samples). May be a  numpy array or a matrix where                  
                m: number of subspaces we are interested in
                T: Number of snapshots of MD trajectory
        
        V -- separating matrix obtained after doing the PCA analysis on m components
             of real data followed by temporal decorrelation of the spatially whitened 
             data
        
        lag -- lag time in the form of an integer denoting by time steps
        
        verbose -- print information on progress. Default is True.
    
    Returns:
    
       Btd4 -- separating matrix                 
        """
    
    # GB: we do some checking of the input arguments and copy data to new
    # variables to avoid messing with the original input. We also require double
    # precision (float64) and a numpy matrix type for X.
    
    warnings.simplefilter("ignore", np.ComplexWarning)
    [n,T] = Z.shape # GB: n is number of input signals, T is number of samples
    
    if m==None:
        m=n 	# Number of sources defaults to # of sensors
    assert m<=n,\
        "TD4 -> Do not ask more sources (%d) than sensors (%d )here!!!" % (m,n)
        
    print "4th order Temporal Decorrelation -> Estimating cumulant matrices"
    
    # Reshaping of the data, hoping to speed up things a little bit...
    
    Z = Z.T
    #Z = Z - Z.mean()
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
    
    # Removing 4th order temporal decorrelations with time delay = lag
    for im in range(m):
        
        Xim = Z[:,im]
        Xijm = multiply(Xim[0 : T - lag :] , Xim[lag : T : ])
             
        # Note to myself: the -R on next line can be removed: it does not affect
        # the joint diagonalization criterion
        
        Qij = dot(multiply(Xijm , Z[0: T-lag :]).T  , Z[lag : T :]) / float(T - lag)  - R - 2 * dot(R[:,im], R[:,im].T)
        
        # To ensure symmetricity of the covariance matrix a mathematical  computation is done
        Qij = (Qij + Qij.T)/2
        CM[:,Range]	= Qij    
        Range = Range  + m 
        for jm in range(im):
                Xijm = multiply(Z[0 : T - lag :, im] , Z[lag : T  :, jm ])
                Qij = sqrt(2) * dot(multiply(Xijm, Z[0 :T - lag : ]).T , Z[lag : T :]) / float(T - lag) \
                    - R[:,im] * R[:,jm].T - R[:,jm] * R[:,im].T
                
                # To ensure symmetricity of the covariance matrix a mathematical  computation is done
                Qij = (Qij + Qij.T)/2
                CM[:,Range]	= Qij
                Range = Range + m 
    
    
    nbcm = int(nbcm)
    # Now we have nbcm = m(m+1)/2 cumulants matrices stored in a big m x m*nbcm array.
    H = matrix(eye(m, dtype=float64))
        
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
        Hkeep = H
      
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
                    H[:,pair] = H[:,pair] * G
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
    
    Btd4 = H.T * V
    
    # Permute the rows of the separating matrix B to get the most energetic components first.
    # Here the **signals** are normalized to unit variance.  Therefore, the sort is
    # according to the norm of the columns of A = pinv(Btd4)

    if verbose:
        print >> stdout, "TD4 -> Sorting the components"
    
    A = pinv(Btd4)
    keys =  array(argsort(multiply(A,A).sum(axis=0)[0]))[0]
    Btd4 = Btd4[keys,:]
    Btd4 = Btd4[::-1,:]     # % Is this smart ?
    
    
    if verbose:
        print >> stdout, "TD4 -> Fixing the signs"
    b	= Btd4[:,0]
    signs = array(sign(sign(b)+0.1).T)[0] # just a trick to deal with sign=0
    Btd4 = diag(signs) * Btd4
    print Btd4.shape
    return Btd4

    """
    
    Main references:
    @article{Special Section on BSS: Independent Component Analysis and Signal Separation
      title 	= "Robust Independent Component Analysis via Time-Delayed Cumulant Functions",
      author       = "Pando Georgiev and Andrzej Cichocki",
      HTML 	= "http://search.ieice.org/bin/summary.php?id=e86-a_3_573",
      journal      = "IEE Proceedings-F",
      month = march, number = 3, pages = {573-580}, volume = E-86-A, year = 2003}
      
      Notes:
      ** Here we consider signals which are spatially and temporally decorrelated of 
      order 2
      
      1) Blind Source Separation problem of signals which are temporally correlated of 
      order 4 can be converted to a symmetric eigenvalue problem of generalized cumulant 
      matrix CM. The measure of independence corresponds to the 'diagonality' of a set of cumulant matrices. 
      Joint diagonalization of these cumulant matrices is done depending on time delay.
      
      2) We expect to use small number of significant eigenmatrices to efficiently 
      summarize all the 4th order information. 
      
      3) Since we are dealing with real signals, it becomes easy to exploit the 
      symmetries of the cumulants to further reduce the number of matrices to be 
      diagonalized.
      
      4) Computational errors can destroy the symmetricity of the cumulant matrices Qij and thus it's restored by performing
      Qij = 0.5*[Qij + Qij.T]
      
      5) The rows of the separating matrix Btd4 are resorted in such a way that the
      columns of the corresponding mixing matrix A=pinv(Btd4) are in decreasing order of (Euclidian)
      norm.  This is a simple, `almost canonical" way of fixing the indetermination of
      permutation.  It has the effect that the first rows of the recovered signals (ie the
      first rows of S = Btd4*data) correspond to the most energetic *components*.  Recall however that
      the source signals in S=Btd4 * data have unit variance.  Therefore, when we say that the
      observations are unmixed in order of decreasing energy, this energetic signature is to
      be found as the norm of the columns of A=pinv(Btd4) and not as the variances of the
      separated source signals.
      
      """
