from __future__ import division
import numpy as np
from scipy import misc, io
from scipy import sparse as sp
from dncuts import dncuts, ncuts
import matplotlib.pyplot as plt
import argparse
import scipy.sparse as sp
import logging
import yaml
import os
import sys

#================================================================================================================================
# This is the driver code behind dncuts. In main() you'll find the image collection, where you can tailor the code to your image.
#================================================================================================================================
 
log = logging.getLogger('main');
log.setLevel(logging.DEBUG);

#Little code to save sparse csc matrices
def save_sparse_csc(filename,array):
    np.savez(filename, data = array.data, indices=array.indices, indptr=array.indptr, shape=array.shape )
#End of save_sparse_csc

#   Loads sparse csc matrices
def load_csc(filename):
    aff_dict = np.load(filename);
    A = sp.csc_matrix((aff_dict['data'], aff_dict['indices'], aff_dict['indptr']), shape=aff_dict['shape']);
    return A;

#Equivalent to Matlabs 'ismember' but only computes indices
def ismember(a,b):
    c = -np.ones(len(a))
    for i in range(len(a)):
        for j in range(len(b)):
            if c[i] != -1:
                continue
            if a[i] == b[j]:
                c[i] = j
    c[np.where(c == -1)] = 0
    return c
#End of ismember

#Compares the true eigens with fast eigens=======================================================================
def grapheigens(evfast, evalfast, evreal, evalreal, nvec=10):
    fig = plt.figure()
    eigenvalues = fig.add_subplot(111)
    eigenvalues.plot(np.arange(nvec), evalfast[:10], 'b-', np.arange(nvec), evalreal[:10], 'r-')
    eigenvalues.set_title('Eigenvalue comparison')
    eigenvalues.legend(('Blue = DNCuts', 'Red = NCuts'), loc='upper left')
    plt.show()
    fig1 = plt.figure()
    fig1.add_subplot(nvec,1,1)
        
    for i in range(nvec):
        plt.subplot(nvec,1,i+1)
        plt.yticks([])
        plt.xticks([])
        plt.plot(np.arange(len(evfast[:,i])), evfast[:,i], 'b-', np.arange(len(evreal[:,i])), evreal[:,i], 'r-')
        if i == nvec-1: plt.xticks(np.linspace(0,len(evfast[:,i]),1000))
    plt.suptitle('Eigenvector comparison')
    plt.legend(('Blue = DNCuts', 'Red = NCuts'), loc='lower right')
    plt.show()
#End of grapheigens =============================================================================================

#Code to visualize the eigenvectors, true and fast ==============================================================
def visualize(evf, evt, evl, im, config):
    nvec = config['n_eigv'];
    log.info('Visualizing...');
    vistrue = evt.reshape(len(im[:,0,0]), len(im[0,:,0]), -1)
    visfast = evf.reshape(len(im[:,0,0]), len(im[0,:,0]), -1)
    vist = vistrue[:,:,:nvec]
    visf = visfast[:,:,:nvec]
    
    vistrue = 4 * np.sign(vist) * np.abs(vist)**(1/2)
    visfast = 4 * np.sign(visf) * np.abs(visf)**(1/2)

    vistrue = np.maximum(0, np.minimum(1, vistrue))
    visfast = np.maximum(0, np.minimum(1, visfast))
    g,h,l = vistrue.shape
    m = np.floor(np.sqrt(l))
    n = np.ceil(l/m)
    mont_true = np.zeros((g*m, h*n))
    mont_fast = np.zeros((g*m, h*n))

    #Construct montage
    count = 0
    for i in range(m.astype(int)):
        for j in range(n.astype(int)):
            try:
                mont_true[i*g:g+i*g,j*h:h+j*h] = vistrue[:,:,count] 
                mont_fast[i*g:g+i*g,j*h:h+j*h] = visfast[:,:,count]
            except:
                mont_true[i*g:g+i*g,j*h:h+j*h] = 0 
                mont_fast[i*g:g+i*g,j*h:h+j*h] = 0
            count = count + 1
    fig = plt.figure();
    ax1 = fig.add_subplot(2,1,1);
    ax1.set_title('True eigenvectors: 1 - {0}'.format(nvec));
    ax1.imshow(mont_true);
    ax2 = fig.add_subplot(2,1,2);
    ax2.set_title('Fast eigenvectors: 1 - {0}'.format(nvec));
    ax2.imshow(mont_fast);
    
    plt.savefig(os.path.join(config['figDir'], 'true_vs_fast_montage.png'));
    pickle.dump(fig, file(path.join(config['figDir'], 'true_vs_fast_montage.pickle'), 'w') );

    #   Plots montage
    if 'graph' in config and config['graph']:
        plt.show();
    plt.clf();

    ax = fig.gca();    
    im = np.zeros((im.shape[0], im.shape[1]))
    ev = visf
    ev = ev.transpose(1,0,2)
    for i in range(nvec):
        im = im + ev[:,:,i]
    ax.imshow(im)

    plt.savefig(os.path.join(config['figDir'], 'eigvonimage.png'));
    pickle.dump(fig, file(path.join(config['figDir'], 'eigvonimage.pickle'), 'w') );

    #   Plots eigenvectors
    if 'graph' in config and config['graph']:
        plt.show();
    plt.clf();

#End of visualize ===============================================================================================

#================================================
#   -Main-
#   Code to construct a gaussian affinity matrix of 'lena.bmp', perform DNCuts, and print eigen-v's
#================================================
def main(config):
    xy = 7 #radius of search
    rgb_sigma = 30 #divide rgb differences by this
    nvec = config['n_eigv'];
    ndown = 2
    saveDir = config['saveDir'];

    require = ['ncut','comp',];
    for req in require:
        if req not in config:
            config[req] = False;

    if 'image' in config:
        #Import image and resize
        log.info('Importing and sizing {0}'.format(config['image']));
        img = misc.imread(config['image'])
        img = misc.imresize(img, tuple(config['size']));

        #Get the pixel affinity matrix and save it
        log.info('Constructing gaussian affinity matrix...');
        A = getgauss(img);
        log.info('Gaussian affinity matrix acquired');
        save_sparse_csc(os.path.join(saveDir,'affinity_lena256.npy'), A);

    elif 'aff' in config:
        log.info('Importing {0} as the affinity matrix...'.format(config['aff']));
        A = load_csc(config['aff']);
    
    if 'A' not in locals():
        raise ImportError('Issue getting your affinity matrix ready.  Check your image or affinity matrix.');

    files = [];

    feigv, feigval = dncuts(A, config);
    files.append(feigv);
    files.append(feigval);
    filenames = ['fast_eigv.npy', 'fast_eigval.npy'];

    if config['ncut'] or config['comp']:
        teigv, teigval = ncuts(A, config);
        files.extend([teigv,teigval,]);
        filenames.extend(['real_eigv.npy', 'real_eigval.npy',]);

    if 'name' in config:
        for i in range(len(filenames)):
           filenames[i] = config['name']+'_'+filenames[i];

    for i in range(len(files)):
        np.save(os.path.join(saveDir, filenames[i]), files[i]);
        
#Potential to call grapheigens
    if 'comp' in config and config['comp']:
        if 'graph' in config and config['graph']: 
            log.info('Comparison before eigenvector processing on first 10 Eigens...');
            grapheigens(feigv, feigval, teigv, teigval, nvec=10);
    #Eigenvector clean up (reordering, resigning...)
        log.info('Cleaning up Eigenvectors...');
        EV_fast = feigv
        EV_true = teigv
        C = np.abs(np.dot(EV_fast.T, EV_true))
        accuracy = np.trace(C)/nvec
        log.info('Accuracy of Eigenvector intially: {0}%'.format(accuracy*100));
        M = np.arange(len(C[:,0]))
        for p in range(10):
            M_last = M
            for i in range(len(C[:,0])):
                for j in range(i+1,len(C[:,0])):
                    if (C[i,M[j]] + C[j,M[i]]) >  (C[i,M[i]] + C[j,M[j]]):
                        m = M[j]
                        M[j] = M[i]
                        M[i] = m
            if np.all(M == M_last):
                break
        M = ismember(np.arange(nvec, dtype=int), M.astype(int))
        
        EV_fast = EV_fast[:,M.astype(int)]
    
        sig = np.sign(np.sum(EV_fast*EV_true, 0))
        EV_fast = EV_fast*sig
        C = np.dot(EV_fast.transpose(), EV_true)
        accuracy = np.trace(C)/nvec
        log.info('Accuracy of Eigenvector after processing: {0}%'.format(accuracy*100));
        np.save(os.path.join(saveDir, 'fast_processed_eigv.npy', EV_fast))
        np.save(os.path.join(saveDir, 'fast_processed_eigval.npy', feigval))
        if 'graph' in config and config['graph']:
            log.info('Comparison after eigenvector processing...');
            grapheigens(EV_fast, feigval, teigv, teigval, nvec=10)
        
        visualize(EV_fast, EV_true, feigval, img, nvec)

#End of Main ====================================================================================================

def getgauss(im, xyrad=7, RGB_SIGMA=30):
    
    g, h, z = im.shape

    # Find all pairs of pixels within a distance of XY_RADIUS
    dj, di = np.meshgrid(np.arange(-xyrad, xyrad + 1), np.arange(-xyrad, xyrad + 1))
    dv = (dj**2 + di**2) <= xyrad**2
    
    di = di[dv]
    dj = dj[dv]
    
    i,j = np.meshgrid(np.arange(1,g+1), np.arange(1,h+1))

    m, n = i.shape
    i = i.reshape(m*n, 1)    
    i = np.tile(i, (1, len(di)))
    
    j = j.reshape(m*n, 1)    
    j = np.tile(j, (1, len(di)))
    
    itemp = i + di.transpose()
    jtemp = j + dj.transpose()
    vtmp = (itemp >= 1) & (itemp <= g) & (jtemp >= 1) & (jtemp <= h)
    
    helper = np.arange(g*h).reshape(g,h)

    pair_i = helper[i[vtmp]-1, j[vtmp]-1]
    pair_j = helper[itemp[vtmp]-1, jtemp[vtmp]-1]
    
    # Weight each pair by the difference in RGB values, divided by RGB_SIGMA
    im = im.transpose(1,0,2)    
    RGB = im.reshape(-1,im.shape[2])/RGB_SIGMA
    w = np.exp(-np.sum((RGB[pair_i,:] - RGB[pair_j,:])**2.0,1))
    #Construct an affinity matrix
    A = sp.csc_matrix((w, (pair_i, pair_j)), shape=(g*h, g*h))
    return A

#================================================
#   Validate:
#   
#   Validates our config file, making directories if necessary
#================================================
def validate(config):
    
    if 'image' in config:
        if not 'size' in config:
            raise ValueError(   'You didn\'t enter the desired size of your image. '+\
                                'Please enter it in the yaml file as detailed in the readme.'
                            );
        elif not type(config['size']) == list:
            raise ValueError(   'You didn\'t enter a valid format for the size for your image. '+\
                                'Please enter it in the yaml file as detailed in the readme.'
                            );
        if 'aff' in config:
            log.warning('You specified an image and an affinity matrix.  Using the image due to precedence.');
    elif not 'aff' in config:
        raise ValueError('You didn\'t enter an image or an affinity matrix! What am I supposed to do?');

    required = ['numblock', 'n_eigv', 'logfile', 'saveDir', 'figDir',];
    for field in required:
        if not field in config:
            raise ValueError('You didn\'t provide a value for the field: {0}'.format(field));

    directories = ['saveDir', 'figDir',];
    for directory in directories:
        if not os.path.isdir(config[directory]):
            os.makedirs(config[directory]);

    if not os.path.isdir(os.path.join(config['saveDir'], '.memmapped')):
        os.makedirs(os.path.join(config['saveDir'], '.memmapped'));
    if not os.path.isdir(os.path.join(config['saveDir'], '.memmapped', 'spmultiply')):
        os.makedirs(os.path.join(config['saveDir'], '.memmapped', 'spmultiply'));


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', action='store_true', dest='graph', default=False, 
                        help='Shows graph of Affinity Matrix and eigenv\'s, depending on flags.');
    parser.add_argument('-n', '--ncut', action='store_true', dest='ncut', default=False, help='Performs NCuts instead of DNCuts.')
    parser.add_argument('-v', action='store_true', dest='verbose', default=False, help='Runs program verbosely.')
    parser.add_argument('-c', '--comp', '--compare', action='store_true', dest='compare', default=False,
                        help='Compares fast and true eigenv\'s.');
    parser.add_argument('-d', '--debug', action='store_true', dest='debug', default=False, help='Prints debugging help.')
    parser.add_argument('--config', type=str, dest='configpath', default='config.yaml',
                        help='Input other configuration file.');
    values = parser.parse_args()

    #   Get config from file
    with open(values.configpath) as f:
        conf_file = f.read();
        config = yaml.load(conf_file);
    if not 'config' in locals(): raise IOError(
    'Issue opening and reading configuration file: {0}'.format(os.path.abspath(values.configpath)) );

    validate(config);

    #   Update config with CLARGS
    level = 30;
    if values.verbose: level = 20;
    elif values.debug: level = 10;
    config['graph'] = values.graph;
    config['ncut'] = values.ncut;
    config['comp'] = values.compare;

    #   Setup stream logger
    ch = logging.StreamHandler(sys.stdout);
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s');
    ch.setLevel(level);
    ch.setFormatter(formatter);

    log.addHandler(ch);

    log.debug('Configuration File:\n'+conf_file);
    log.info('Using Configuration File: {0}'.format(os.path.abspath(values.configpath)));

    main(config);
