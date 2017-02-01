from __future__ import division
import numpy as np
from numpy import *
from scipy.sparse.linalg import *
import scipy.sparse as sp
from scipy.linalg import eig
import matplotlib.pyplot as plt
from _norm_c import *
import sys
import logging
import os

log = logging.getLogger('main.dncuts');
log.setLevel(logging.DEBUG);

#==============================================================================
# This is a python translation of 'Downsampled Normalized Cuts', an algorithm
# created by researchers at UC Berkely for image segmentation.
#==============================================================================
def save_sparse_csc(filename,array):
    np.savez(filename, data = array.data, indices=array.indices, indptr=array.indptr, shape=array.shape )

def to_array(a):
    shape = 0;
    for i in a: shape += len(i);
    array = np.zeros(shape, dtype='int64');
    
    c = 0;
    for i in range(len(a)):
        temp = np.array(a[i]);
        array[c:c+temp.shape[0]] = temp;
        c += temp.shape[0];
    
    return array;

def sparse_truncate(a, thresh):
    a.data = np.multiply((a.data > thresh),a.data);
    a.eliminate_zeros();

def update_progress(progress):
    barLength = 40 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

"""#================================================
   sparse_multiply

    Function to multiply large sparse matrices 
    w/o memory errors and print density
    **Note** - This is really quite a slow function,
    but on machines where the data matrices are too
    large initially, it allows you to still compute
    the eigenvectors / values. (I was also using
    4GB RAM + 4GB Swap when I had to use this though)

    Set numblock to 1 to effectively ignore this code.

    We want n % 2 = 0 because if we can handle
    multiplying shape [a/2n,a]*[a,a/2n] matrices, 
    i.e. ( (a^2)/4(n^2) dot products in memory), then on the next run,
    we can still handle that amount, so we decrease
    our number of blocks (increasing speed), to hit that
    level i.e. ( (a^2)/16(x^2) dot products bc of downsampling),
    x = n / 2 to equate number of in RAM dot products.
#================================================"""
def sparse_multiply(a,b,n, config):
    assert( a.getformat() == 'csr' );
    assert( b.getformat() == 'csc' );
    saveDir = config['saveDir'];
    c = [];
    #    a = [a_1 ... a_n]^T    
    data = [];
    indices = [];
    indptr = [];
    submat_shape = np.array([a.shape[0]//n,b.shape[1]//n]);
    metarow = [];
    log.info('Beginning Multiplication');
    for i in range(n):
        metarowlen = 0;
        for j in range(n):
            a_lower = (a.shape[0]//n)*i;
            a_upper = (a.shape[0]//n)*(i+1);
            if i == n-1: a_upper = a.shape[0];

            b_lower = (b.shape[1]//n)*j;
            b_upper = (b.shape[1]//n)*(j+1);
            if j == n-1: b_upper = b.shape[1];

            a_i = a[a_lower:a_upper];
            b_i = b[:, b_lower:b_upper];

            c = sp.csr_matrix.dot(a_i,b_i);
            #   TL;DR -- Bad things happen if c is trivial, this won't affect eigenvectors
            if not len(c.data):
                c[0,0] = 0.00002;
            shp = len(c.data);
            metarowlen += shp;

            sparse_truncate(c[-1], 0.00001);    #    In-method truncation to limit dense construction

            sub_ind = np.memmap(
                                os.path.join(
                        saveDir,
                        '.memmapped/spmultiply/{0}_{1}indices.array'.format(i,j)
                                            ),
                                dtype='int64',
                                mode='w+',
                                shape=shp,
                               );
            sub_data = np.memmap(
                                os.path.join(
                        saveDir,
                        '.memmapped/spmultiply/{0}_{1}data.array'.format(i,j),
                                            ),

                                dtype='float64',
                                mode='w+',
                                shape=shp,
                                );
            sub_indptr = np.memmap(
                                os.path.join(
                        saveDir,
                        '.memmapped/spmultiply/{0}_{1}indptr.array'.format(i,j),
                                            ),
                                dtype='int64',
                                mode='w+',
                                shape=(submat_shape[0]+1),
                                  );
            sub_ind[:] = c.indices.astype('int64');
            sub_data[:] = c.data.astype('float64');
            sub_indptr[:] = c.indptr.astype('int64');
            sub_ind.flush(); sub_data.flush(); sub_indptr.flush();
            del c, sub_ind, sub_data, sub_indptr;
            update_progress( (j+i*n+1) / float(n**2) );
        metarow.append(metarowlen);
    log.info('Multiplication complete, beginning reassembly.');
    #    Reassemble
    metaindptr = [];
    start = 0;
    end = 0;
    ind = np.memmap(
            os.path.join(
                saveDir,
                '.memmapped/spmultiply/metarowindices.array',
                        ),
            dtype='int64',
            mode='w+',
            shape=np.sum(np.array(metarow)),
                   );
    data = np.memmap(
            os.path.join(
                saveDir,
                '.memmapped/spmultiply/metarowdata.array',
                        ),
            dtype='float64',
            mode='w+',
            shape=np.sum(np.array(metarow)),
                    );
    for i in range(n):        
        indptr = [];
        indptr.append(0);
        for k in range(submat_shape[0]):
            rowlen = 0;
            for j in range(n):
                start = end;
                sub_ind = np.memmap(
                    os.path.join(
                        saveDir,
                        '.memmapped/spmultiply/{0}_{1}indices.array'.format(i,j),
                                ),
                    dtype='int64',
                    mode='r+',
                                   );
                sub_data = np.memmap(
                    os.path.join(
                        saveDir,
                        '.memmapped/spmultiply/{0}_{1}data.array'.format(i,j),
                                ),
                    dtype='float64',
                    mode='r+',
                                   );
                sub_indptr = np.memmap(
                    os.path.join(
                        saveDir,
                        '.memmapped/spmultiply/{0}_{1}indptr.array'.format(i,j),
                                ),
                    dtype='int64',
                    mode='r+',
                                   );                
                rowdata = sub_data[sub_indptr[k]:sub_indptr[k+1]];
                rowind = sub_ind[sub_indptr[k]:sub_indptr[k+1]];
                
                datalen = rowdata.shape[0];
                end = start + datalen;
                rowlen += datalen;

                ind[start:end] = (rowind+submat_shape[1]*j).astype('int64');
                
                data[start:end] = rowdata.astype('float64');
                update_progress( (j+k*n+n*i*(submat_shape[0])+1) / float((n**2)*submat_shape[0]) );
            indptr.append(indptr[-1]+rowlen);
        if i == 0: metaindptr.append(indptr);
        else: metaindptr.append(metaindptr[-1][-1]+np.array(indptr[1:]));
        ind.flush(); data.flush();
    
    indptr = to_array(metaindptr);    #supplementary function
    aff = sp.csc_matrix((data[:], ind[:], indptr.astype('int64')), shape=(a.shape[0],a.shape[0]));
    del sub_ind, sub_data, sub_indptr, rowdata, rowind, indptr, metaindptr, b_i, a_i, a,b;
    return aff;

def half_sparse(a):
    ind = np.memmap('tempindices.array', dtype='int64', mode ='w+', shape=a.indices.shape);
    indptr = np.memmap('tempindptr.array', dtype='int64', mode ='w+', shape=np.ceil(a.shape[0]/2)+1);
    data = np.memmap('tempdata.array', dtype='float64', mode ='w+', shape=a.data.shape);
    bottom = 0;
    indptr[0] = np.float64(0);
    for i in range(0,a.indptr.shape[0]-a.indptr.shape[0]%2,2):
        di = a.indptr[i+1] - a.indptr[i];
        #print di;
        top = bottom + di;
        ind[bottom:top] = a.indices[a.indptr[i]:a.indptr[i+1]];
        data[bottom:top] = a.data[a.indptr[i]:a.indptr[i+1]];
        indptr[i//2+1] = np.float64(top);
        bottom = top;
    #return sp.csc_matrix( (data[:top], ind[:top], indptr[:]), shape=(a.shape[1],np.ceil(a.shape[0]/2)) )
    return sp.csr_matrix( (data[:top], ind[:top], indptr[:]), shape=(np.ceil(a.shape[0]/2),a.shape[1]) )

#================================================
# Whitens Eigenvectors
#================================================

def whiten(x):
    xcent = x - mean(x,0)
    c = np.dot(transpose(xcent),xcent)
    
    D, V = eig(c)
    iD = diag(np.sqrt(1.0/D))
    trans = np.dot(np.dot(V, iD),V.transpose())
    
    xwhite = np.dot(xcent, trans)
    return xwhite

#===========================================================================
# 'Normalized Cuts', finds k non-zero eigenvectors of matrix and normalizes.
#===========================================================================
def ncuts(A, k=16):
    assert( A.shape[0] == A.shape[1] );
    n = A.shape[0];
    offset = .5;    

    d = np.array(A.sum(1) + 2*offset);
    dr = sp.diags(np.ones(n)*offset,0,format='csc');
    A = A + dr;
    dsqinv = (1./ np.sqrt(d  + np.spacing(1))).flatten();

    P = sp.diags(dsqinv, 0, format='csc').dot(A).dot(sp.diags(dsqinv, 0, format='csc'));

    log.info('Solving for eigenvalues and eigenvectors...');
    
    Eval, Ev = eigsh(P, k=k);

    log.info('Solved!');

    #Sort vectors in descending order, leaving out the zero vector
    idx = np.argsort(-Eval)
    Ev = Ev[:,idx].real
    Eval = Eval[idx].real
    
    #Make vectors unit norm
    for i in range(k):
        Ev[:,i] /= np.linalg.norm(Ev[:,i]);

    return Eval, Ev;

#   Small Helper function        
def trunc(data, indices, indptr, n):

    indptr = indptr[:n+1];
    indices = indices[:indptr[-1]];
    data = data[:indptr[-1]];
    return data, indices, indptr;

#   Brings matrix shape down to [n x n]
def trunc_sp_matrix(a, n):
    if not a.getformat() == 'csc':
        a = sp.csc_matrix(a);

    shape  = a.shape;
    data, indices, indptr = trunc(a.data, a.indices, a.indptr);

    a = sp.csc_matrix((data, indices, indptr), shape=(n, shape[1])).tocsr();

    shape  = a.shape;
    data, indices, indptr = trunc(a.data, a.indices, a.indptr);

    return sp.csr_matrix((data, indices, indptr), shape=(shape[0], n)).tocsc();
    

#================================================
#   'Downsampled Normalized Cuts', 'downsamples' 
#   similarity matrix and uses NCuts to return 
#   'upsampled' normalized eigenvectors and 
#   eigenvalues.
#================================================
def dncuts(a, config, n_downsample=2, decimate=2):

# a = affinity matrix
# nevc = number of eigenvectors (set to 16?)
# n_downsample = number of downsampling operations (2 seems okay)
# decimate = amount of decimation for each downsampling operation (set to 2)
    
    nvec = config['n_eigv'];
    a_down=sp.csc_matrix(a);
    n = np.abs(config['numblock']);        #    Blocks in sparse multiplication
    #   N needs to be divisible by 4 bc. of sparse multiplication.
    if n % 2 != 0:
        n = 2*(n // 2);
        if n == 0:
            n = 1;
        log.warning('You gave a \'numblock\' value ({0}) which wasn\'t a multiple of 2. '.format(config['numblock'])+\
                    'It is being treated as ({0}) now.'.format(n)
                   );
    if a_down.shape[0] % 2*n != 0:
        a_down = trunc_sp_matrix(a_down, 2*n*(a_down.shape[0] // 2*n));
    
    
    list1 = [None] * n_downsample
    
    #Loop Start =========================================
    for di in range(n_downsample):
        
#Decimate sparse matrix
        a_sub = half_sparse(a_down);
        log.debug('a_sub shape: {0}'.format(a_sub.shape));
        log.info('Downsampled sparse matrix #{0}'.format(di+1));
        #a_sub = a_sub.transpose();

# Normalize the downsampled affinity matrix using C-code parallelized with OpenMP
        log.info('Normalizing matrix...')
        a_tmp = a_sub.tocsc();
        d = array(a_tmp.sum(0)).reshape(a_tmp.shape[1])
        #np.save('savefiles/tmp_indptr.npy', a_tmp.indptr); np.save('savefiles/tmp_data.npy', a_tmp.data);   #   Debugging help
        norm(d.size, d, a_tmp.indptr, a_tmp.data)
        log.info('Matrix Normalized')
        b = a_tmp.transpose().tocsc();

# "Square" the affinity matrix, while downsampling
        log.debug('a_sub shape: {0}'.format(a_sub.shape));
        log.debug('b shape: {0}'.format(b.shape));
        log.info('Before dot-product')
        log.debug('number of non zeros in a_sub: {0}'.format(a_sub.nnz));
        log.debug('number of non zeros in b: {0}'.format(b.nnz));
        #print np.sum(a_sub.data > 0.0001);   #   Debugging help
        del a_tmp, d; 
        log.debug('a_sub format: {0}, b format: {1}'.format(a_sub.getformat(), b.getformat()));
        a_down = sparse_multiply(a_sub, b, n, config)
        log.info('After dot-product')
        
        
        #print 'adown', a_down.shape
        #save_sparse_csc('adown%i.npz'%(di), a_down);   #   Debugging help
  
# Hold onto the normalized affinity matrix for upsampling later
        list1[di]=b
        log.info('End of loop #{0}'.format(di+1))
    #Loop end ============================================  
    
# Get the eigenvectors
    del a_sub, b;
    #save_sparse_csc('savefiles/a_down.npz',a_down);   #   Debugging help
    Eval, Ev = ncuts(a_down, k=nvec)

    #np.save('eigenvectors.npy', Ev);   #   Debugging help
    #np.save('eigenvalues.npy', Eval);  #   Debugging help

#Some debugging help    
    #print 'list', list1[0].shape
    #print 'list', list1[1].shape
    #print 'ev', Ev.shape
    #print 'eval', Eval.shape, Eval

# Upsample the eigenvectors
    log.info('Upsampling eigenvectors...')
    for di in range(n_downsample-1,-1,-1):
        Ev = list1[di].dot(Ev)
        #print 'ev', Ev.shape   #   Debugging help
    log.info('Eigenvectors upsampled')

# "Upsample" the eigenvalues
    #print Eval   #   Debugging help
    Eval = 1./(2.**(n_downsample))*Eval
    #print Eval   #   Debugging help
    log.info('\'Upsampled\' eigenvalues')

# whiten the eigenvectors
    log.info('Whitening eigenvectors...')
    Ev = whiten(Ev);
    log.info('Eigenvectors whitened')
    
    return Eval, Ev;
