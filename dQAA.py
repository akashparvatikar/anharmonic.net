import numpy
import numpy as np
import math
import scipy.stats
from scipy.stats import kurtosis
from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
 
import argparse
import os
import logging
import pickle

from KabschAlign import *
from IterativeMeansAlign import *
from jade import *

from MDAnalysis import Universe
import MDAnalysis.version as v


mdversion = v.__version__;
tmp = '';
b = False;
for i in mdversion:
    if i == '.': b = not(b);
    if b: tmp += i;
tmp = tmp.strip('.');
assert( int(tmp) >= 11 );

log = logging.getLogger('main.cQAA');
log.setLevel(logging.DEBUG);

#================================================
#a is mem-mapped array, b is array in RAM we are adding to a.
def mmap_concat(shape,b,filename):
    assert(shape[1] == b.shape[1] and shape[2] == b.shape[2]);
    c = np.memmap(filename, dtype='float64', mode='r+', shape=(shape[0]+b.shape[0],shape[1],shape[2]))
    c[shape[0]:, :, : ] = b
    newshape = c.shape;    
    del c;    #    Flushes changes to memory and then deletes.  New array can just be called using filename
    return newshape;
#================================================

def phisel(res, resid):
#MDAnalysis' phi_selection requires a segid be present while this doesn't.
	return res[resid-1].C + res[resid].N + res[resid].CA + res[resid].C

def psisel(res, resid):
#MDAnalysis' phi_selection requires a segid be present while this doesn't. 
	return res[resid].N + res[resid].CA + res[resid].C + res[resid+1].N

def dqaa(config, val):

    #    Pull config from config
    startRes = config['startRes'];
    endRes = config['endRes']+1;
    sliceVal = config['sliceVal'];
    trajectories = config['trajectories'];
    savedir = config['saveDir'];
    figdir = config['figDir'];

    filename = os.path.join(savedir,'.memmapped','coord_data.array');

	#	Loops through trajectories --------------------------------------------
    count = 0;
	for traj in trajectories:
        count+=1;
		#	!Edit to your trajectory format!
		try:
			pdb = config['pdbfile'];
            dcd = traj;
            u = Universe(pdb, dcd, permissive=False);
		except:
            log.debug('PDB: {0}\nDCD: {1}'.format(pdb, dcd));
            raise ImportError('You must edit \'config.yaml\' to fit your trajectory format!');

        protein_residues = u.n_residues;
        
	
		phidat = TimeseriesCollection()
		psidat = TimeseriesCollection()

		#	Adds each (wanted) residues phi/psi angles to their respective timeseries collections
		log.info('Processing Trajectory {0}...'.format(count));

        start   = np.max( [startRes, 1] );
        end     = np.min( [endRes, protein_residues - 1] );

        resnames = u.atoms.CA.resnames[start:end];        

        numRes  = end - start + 1;
		for res in range(start, end):
			#	selection of the atoms involved for the phi angle
			phi_sel = phisel(u.residues,res);
			#	selection of the atoms involved for the psi angle
			psi_sel = psisel(u.residues,res);

			phidat.addTimeseries(Timeseries.Dihedral(phi_sel))
			psidat.addTimeseries(Timeseries.Dihedral(psi_sel))

		#	Computes along whole trajectory
		phidat.compute(u.trajectory, skip=config['sliceVal'])
		psidat.compute(u.trajectory, skip=config['sliceVal'])
	
		#	Converts to nd-array and changes from [numRes,1,sliced numSamples] 
        #   to [numRes,sliced numSamples]
		phidat =  array(phidat).reshape(numRes,-1);
		psidat =  array(psidat).reshape(numRes,-1);
		
		"""	Data stored as  | sin( phi_(i) )   |---
							| cos( phi_(i) )   |---
							| sin( psi_(i) ) |---
							| cos( psi_(i) ) |---"""
		didat = np.empty((4*phidat.shape[0], phidat.shape[1]));
		didat[0::4,:] = np.sin(phidat);
		didat[1::4,:] = np.cos(phidat);
		didat[2::4,:] = np.sin(psidat);
		didat[3::4,:] = np.cos(psidat);

		if count == 1:
			fulldat = np.memmap(filename,
                                dtype='float64',
                                mode='w+',
                                shape=(useable_res*4, tmplen),);

			fulldat[:,:] = didat.astype('float64');
			map_shape = fulldat.shape;
			del fulldat;
			del didat;
		else:
			map_shape = mmap_concat(map_shape, didat, filename);
	#	End loop --------------------------------------------------------------
	
	fulldat = np.memmap(filename, dtype='float64', mode='r+', shape=map_shape);
	
    np.save(os.path.join(savedir, '{0}_dihedral_data.npy'.format(config['pname'])), fulldat[:,:]);
    np.save(os.path.join(savedir, '{0}_resnames.npy'.format(config['pname'])), resnames);
    
    del fulldat;

	icajade, icafile, mapshape = jade_calc(config, filename, map_shape);
	return icajade, icafile, mapshape;
#================================================

#================================================
def genCumSum(config, pcas, pcab):
    savedir = config['saveDir'];
    figdir = config['figDir'];

    si = numpy.argsort(-pcas.ravel());
    pcaTmp = pcas;
    pcas = numpy.diag(pcas);
    pcab = pcab[:,si];
    
    fig = plt.figure();
    ax = fig.add_subplot(111);
    y = numpy.cumsum(pcaTmp.ravel()/numpy.sum(pcaTmp.ravel()));
    ax.plot(y*100);
    ax.set_xlabel('Number of Principal Components');
    ax.set_ylabel('Percent of Total Variance Preserved');
    ax.set_title('Variance of Principal Components');

    pickle.dump(fig, file(os.path.join(figdir, '{0}_cumsum.pickle'.format(config['pname'])), 'w') );
    plt.savefig(os.path.join(figdir, '{0}_cumsum.png'.format(config['pname'])));

    if 'setup' in config and config['setup']:
        log.info('Cov. Matrix spectrum cum. sum:');
        plt.show();
        a = input('Enter desired ICA dimension (enter -1 for default): ');
        if (a > 0):
            config['icaDim'] = a;
#================================================

#================================================
def genKurtosisPlot(config, filename, mapshape):
    mapcoords = np.memmap(filename, 'r', dtype='float64', shape=mapshape);

    #   May have to truncate
    numTraj = mapcoords.shape[1] // config['trajLen'];
    plotindv = numTraj <= 5;
    mapcoords = mapcoords[:,:numTraj*config['trajLen']];

    mapcoords[:,:] = mapcoords - np.mean(mapcoords, axis=1).reshape((-1,1));
    
    #   Using Pearson's defn of kurtosis: (Fisher + 3)
    d = mapcoords.flatten();
    mean    = np.mean(d);
    stddev  = np.std(d);
    kurt    = kurtosis(d, 0, fisher=0);

    values, edges = np.histogram(d, bins=51, normed=1);
    gauss_curve = np.exp( - ((edges - mean)**2.)/(2*stddev**2.));
    #   Normalizes such that the integral of gauss_curve over boundary is 1
    gauss_curve = gauss_curve / ( np.sum(gauss_curve) * np.abs(edges[1]-edges[0]) );
    midpts = .5*( edges[1:] + edges[:-1] );

    fig = plt.figure();
    ax = fig.gca();
    #   Log-Plot
    ax.semilogy(midpts, values, 'r-', label='{0} Data, Kurtosis: {1}'.format(config['pname'], kurt));
    #   gauss curve not very useful
    #ax.semilogy(edges, gauss_curve, 'b-', label='Gaussian Curve, Kurtosis: 3');

    if plotindv:
        kurtdata = [];
        scale = mapcoords.shape[0]*config['trajLen'];
        for i in range(numTraj):
            kurtdata.append(kurtosis(d[i*scale:(i+1)*scale], 0, fisher=False));

        val, edg, mid = [], [], [];
        for i in range(numTraj):
            a, b = np.histogram(d[i*scale:(i+1)*scale], bins=51, normed=1);
            mpt = (b[1:]+b[:-1])/2.;
            val.append(a);
            edg.append(b);
            mid.append(mpt);

        for i in range(numTraj):
            ax.semilogy( mid[i], val[i], label='{0} Data, Traj. {1},  Kurtosis: {2}'.format(config['pname'], i+1, kurtdata[i]));

    box = ax.get_position();
    ax.set_position([box.x0, box.y0+box.height*.2, box.width, box.height*.8]);
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.1), fancybox=True, fontsize=8);
    ax.set_title('Plotted Histogram of {0} data (Log y-scale)'.format(config['pname']));

    plt.savefig( os.path.join(config['figDir'], '{0}_loghistogram.png'.format(config['pname'])) );
    pickle.dump( fig, file( os.path.join(config['figDir'], '{0}_loghistogram.pickle'.format(config['pname'])), 'w+') );

    if 'graph' in config and config['graph']:
        plt.show();

    plt.close();


#================================================

#================================================
def jade_calc(config, filename, mapshape):
    dim = 3;
    numres = mapshape[0]/dim;
    numsamp = mapshape[1];
    coords = np.memmap(filename, dtype='float64', shape=mapshape);
    savedir = config['saveDir'];
    figdir = config['figDir'];

    #    Plots (if `setup`) Cum. Sum of Variance in PC's
    [pcas,pcab] = numpy.linalg.eig(numpy.cov(coords));
    genCumSum(config, pcas, pcab);

    #   Plots Histogram
    genKurtosisPlot( config, filename, mapshape );

    # some set up for running JADE
    subspace = config['icaDim']; # number of IC's to find (dimension of PCA subspace)
    
    #    Performs jade and saves if main
    coords.flush();
    icajade = jadeR(filename, mapshape, subspace);
    np.save(os.path.join(savedir, '{0}_icajade_{1}dim.npy'.format(config['pname'], config['icaDim'])), icajade); 
    log.debug('icajade: {0}'.format(numpy.shape(icajade)));

    #    Performs change of basis
    icafile = os.path.join(savedir,'.memmapped','icacoffs.array');
    icacoffs = np.memmap(icafile, dtype='float64', mode='w+', shape=(config['icaDim'],numsamp) );
    icacoffs[:,:] = icajade.dot(coords)
    icacoffs.flush();

    #   Pulls it out of the hidden directory for later use
    public_icafile = os.path.join(savedir, '{0}_icacoffs_{1}dim.array'.format(config['pname'],config['icaDim']));
    publicica = np.memmap(public_icafile, dtype='float64', mode='w+', shape=(config['icaDim'],numsamp) );
    publicica[:,:] = icacoffs[:,:];
    publicica.flush();    
    del publicica;

    log.debug('icacoffs: {0}'.format(numpy.shape(icacoffs)));
    np.save(os.path.join(savedir, '{0}_icacoffs_{1}dim.npy'.format(config['pname'], config['icaDim'])), icacoffs[:,:]) 

    fig = plt.figure();
    ax = fig.add_subplot(111, projection='3d');
    ax.scatter(icacoffs[0,::10], icacoffs[1,::10], icacoffs[2,::10], marker='o', c=[0.6,0.6,0.6]);

    #    Saves figure object to pickle, saves figure plot to png
    pickle.dump(fig, file(os.path.join(figdir, '{0}_icacoffs_scatter.pickle'.format(config['pname'])), 'w') );
    plt.savefig(os.path.join(figdir, '{0}_icacoffs_scatter.png'.format(config['pname'])));
    
    #    If in CLI and graph flag
    if 'graph' in config and config['graph']:
        log.info('First 3 Dimensions of Icacoffs');
        plt.show();

    #    Returns icajade matrix, ica filename, and the shape.
    return icajade, icafile, icacoffs.shape;
#================================================

#================================================
"""
    INPUT:
    fulldat     = atom coordinates      -- size [NumAtoms, 3, NumRes];
    atomname    = list of atom names    -- length NumRes
    resname     = list of residuee names-- length NumRes
    filename    = filename to save to: e.g. 'filename.pdb'
    verbose     = boolean to  
"""

def pdbgen(fulldat, atomname, resname, filename, verbose):
    
    numatom, dim, numres = fulldat.shape;
    assert ( len(atomname) == numres );
    assert ( len(resname) == numres );

    if verbose: print 'Constructing PDB file...';
    f = open('savefiles/%s.pdb' %(filename), 'w+');
    for i in range(fulldat.shape[0]):
        f.write('%-6s    %4i\n' %('MODEL', i+1));
        for j in range(fulldat.shape[2]):
            f.write('%-6s%5i %4s %s  %4i    %8.3f%8.3f%8.3f%6.2f%6.2f           C  \n' \
                %('ATOM', j+1, atomname[j], resname[j], j+1, fulldat[i,0,j], fulldat[i,1,j], fulldat[i,2,j], 0.0, 0.0,));
        f.write('ENDMDL\n');
    f.close();
    if verbose: print 'PDB file completed!';

#================================================
