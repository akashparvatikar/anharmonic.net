import numpy
import numpy as np
import math
import scipy.stats
import matplotlib.pyplot as plt
from KabschAlign import *
from IterativeMeansAlign import *
from MDAnalysis.core.Timeseries import *
from MDAnalysis import *
import MDAnalysis.version as v
from numpy import *
from jade import *
import argparse
from scipy.stats import kurtosis
import os.path as path
import logging
import pickle

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
	del c;	#	Flushes changes to memory and then deletes.  New array can just be called using filename
	return newshape;
#================================================


#	Main  Code
#	||		||
#	\/		\/

#================================================
def qaa(config):

	iterAlign = IterativeMeansAlign();
	itr = []; avgCoords = []; eRMSD = [];

	#	Pull config from config
	startRes = config['startRes'];
	endRes = config['endRes'];
	sliceVal = config['sliceVal'];
	trajectories = config['trajectories'];
	savedir = config['saveDir'];
	figdir = config['figDir'];

	global dt;
	dim = 3;
	filename = path.join(savedir,'.memmapped','coord_data.array');

	#----------------------------------------------
	count = 0;
	for traj in trajectories:
		count+=1;
		try:
			pdb = config['pdbfile'];
			dcd = traj;
			u = MDAnalysis.Universe(pdb, dcd, permissive=False);

		except:
			log.debug('PDB: {0}\nDCD: {1}'.format(pdb, dcd));
			raise ImportError('You must edit \'config.yaml\' to fit your trajectory format!');

		atom = u.atoms.CA;
		resnames = list(atom.resnames);
		dt = u.trajectory.dt;

		log.debug('Time Delta: {0}'.format(dt));
		log.info('Processing Trajectory {0}...'.format(count));

		counter = 0;
		cacoords = []; frames = [];

		for ts in u.trajectory:
			if (counter % config['sliceVal'] == 0):
				f = atom.positions;
				cacoords.append(f.T);
				frames.append(ts.frame);
			counter = counter + 1;

		[a, b, c, d] = iterAlign.iterativeMeans(array(cacoords)[:,:,startRes-1:endRes], 0.150, 4);

		#	Keeping data		
		itr.append(a);
		avgCoords.append(b[-1]);
		eRMSD.append(c);
		if count == 1:
			mapped = np.memmap(filename, dtype='float64', mode='w+', shape=d.shape);
			mapped[:,:,:] = d.astype('float64');
			map_shape = mapped.shape;
			del mapped;
			del d;
		else:
			map_shape = mmap_concat(map_shape, d, filename);

		#mapped = np.memmap(filename, dtype='float64', mode='r+', shape=map_shape);
	
	#	END FOR -------------------------------------------- END FOR

	log.debug( 'Saved array shape: {0}'.format(map_shape));

	num_coords = map_shape[0];
	dim = map_shape[1];
	num_atoms = map_shape[2];
	trajlen = len(u.trajectory) / sliceVal

	log.debug( 'num_coords: {0}'.format(num_coords));

	#	Final alignment
	if len(trajectories) > 1:
		[itr, avgCoords, eRMSD, junk ] = iterAlign.iterativeMeans(0, 0.150, 4, mapped=True, fname=filename, shape=map_shape);	

	log.debug( 'eRMSD shape: {0}'.format(numpy.shape(eRMSD)));
	np.save(path.join(savedir, '{0}_eRMSD.npy'.format(config['pname'])), eRMSD );

	#	Import and reshape
	mapalign = np.memmap(filename, dtype='float64', mode='r+', shape=((num_coords,dim,num_atoms)) );
	newfilename = path.join(savedir,'.memmapped','coord.array');
	mapped = np.memmap(newfilename, dtype='float64', mode='w+', shape=((dim*num_atoms, num_coords)));
	for i in range(3): mapped[i::3,:] = mapalign[:,i,:].T;
	
	mapshape = mapped.shape;
	filename=mapped.filename;
	mapped.flush();

	np.save(path.join(savedir, '{0}_coords.npy'.format(config['pname'])), mapped[:,:]);
	np.save(path.join(savedir, '{0}_resnames.npy'.format(config['pname'])), resnames);
	del mapped;
	icajade, icafile, mapshape = jade_calc(config, filename, mapshape);
	return icajade, icafile, mapshape;
#================================================

"""#============================================= (Not memmapped as of yet, don't use)
def minqaa(config, fulldat):
	#	Setup
	iterAlign = IterativeMeansAlign();
	itr = []; avgCoords = []; eRMSD = [];
	dim = 3;
	assert(len(fulldat.shape) == 3);
	num_coords = fulldat.shape[0];
	dim = fulldat.shape[1];
	num_atoms = fulldat.shape[2];
	
	#	Final averaging
	[itr, avgCoords, eRMSD, fulldat ] = iterAlign.iterativeMeans(fulldat, 0.150, 4);	
	
	log.debug( 'eRMSD shape: {0}'.format(numpy.shape(eRMSD)) );
	np.save(path.join(savedir, '{0}_eRMSD.npy'.format(config['pname'])), eRMSD );
	
	#	Reshaping of coords
	coords = fulldat.reshape((fulldat.shape[0],-1), order='F').T

	np.save(path.join(savedir, '{0}_coords.npy'.format(config['pname'])), coords);
	#pdbgen(fulldat, resname);
	jade_calc(config, coords, val, avgCoords, num_coords);

#================================================"""

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

	pickle.dump(fig, file(path.join(figdir, '{0}_cumsum.pickle'.format(config['pname'])), 'w') );
	plt.savefig(path.join(figdir, '{0}_cumsum.png'.format(config['pname'])));

	if 'setup' in config and config['setup']:
		log.info('Cov. Matrix spectrum cum. sum:');
		plt.show();
		a = input('Enter desired ICA dimension (enter -1 for default): ');
		if (a > 0):
			config['icaDim'] = a;
#================================================

#================================================
def jade_calc(config, filename, mapshape):
	dim = 3;
	numres = mapshape[0]/dim;
	numsamp = mapshape[1];
	coords = np.memmap(filename, dtype='float64', shape=mapshape)
	savedir = config['saveDir'];
	figdir = config['figDir'];

	#	Plots (if `setup`) Cum. Sum of Variance in PC's
	[pcas,pcab] = numpy.linalg.eig(numpy.cov(coords));
	genCumSum(config, pcas, pcab);

	# some set up for running JADE
	subspace = config['icaDim']; # number of IC's to find (dimension of PCA subspace)
	
	#	Performs jade and saves if main
	coords.flush();
	icajade = jadeR(filename, mapshape, subspace);
	np.save(path.join(savedir, '{0}_icajade_{1}dim.npy'.format(config['pname'], config['icaDim'])), icajade); 
	log.debug('icajade: {0}'.format(numpy.shape(icajade)));

	#	Performs change of basis
	icafile = path.join(savedir,'.memmapped','icacoffs.array');
	icacoffs = np.memmap(icafile, dtype='float64', mode='w+', shape=(config['icaDim'],numsamp) );
	icacoffs[:,:] = icajade.dot(coords)
	icacoffs.flush();

    #   Pulls it out of the hidden directory for later use
    public_icafile = path.join(savedir, '{0}_icacoffs_{1}dim.array'.format(config['pname'],config['icaDim']));
    publicica = np.memmap(public_icafile, dtype='float64', mode='w+', shape=(config['icaDim'],numsamp) );
	publicica[:,:] = icacoffs[:,:];
    publicica.flush();	
    del publicica;

	log.debug('icacoffs: {0}'.format(numpy.shape(icacoffs)));
	np.save(path.join(savedir, '{0}_icacoffs_{1}dim.npy'.format(config['pname'], config['icaDim'])), icacoffs[:,:]) 

	fig = plt.figure();
	ax = fig.add_subplot(111, projection='3d');
	ax.scatter(icacoffs[0,::10], icacoffs[1,::10], icacoffs[2,::10], marker='o', c=[0.6,0.6,0.6]);

	#	Saves figure object to pickle, saves figure plot to png
	pickle.dump(fig, file(path.join(figdir, '{0}_icacoffs_scatter.pickle'.format(config['pname'])), 'w') );
	plt.savefig(path.join(figdir, '{0}_icacoffs_scatter.png'.format(config['pname'])));
	
	#	If in CLI and graph flag
	if 'graph' in config and config['graph']:
		log.info('First 3 Dimensions of Icacoffs');
		plt.show();

	#	Returns icajade matrix, ica filename, and the shape.
	return icajade, icafile, icacoffs.shape;
#================================================

#================================================
"""
    INPUT:
    fulldat     = atom coordinates      -- size [NumAtoms, 3, NumRes];
    atomname    = list of atom names    -- length NumRes
    resname     = list of residuee names-- length NumRes
    elementname = list of element names -- length NumRes
    filename    = filename to save to: e.g. 'filename.pdb'
    verbose     = boolean to  
"""

def pdbgen(fulldat, atomname, resname, filename, verbose):
	
	numatom, dim, numres = fulldat.shape;
	assert ( len(atomname) == numres );
	assert ( len(resname) == numres );
	assert ( len(elementname) == numres );

	if verbose: print 'Constructing PDB file...';
	f = open('savefiles/%s.pdb' %(filename), 'w+');
	for i in range(fulldat.shape[0]):
		f.write('%-6s    %4i\n' %('MODEL', i+1));
		for j in range(fulldat.shape[2]):
			f.write('%-6s%5i %4s %s  %4i    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s  \n' \
				%('ATOM', j+1, atomname[j], resname[j], j+1, fulldat[i,0,j], fulldat[i,1,j], fulldat[i,2,j], 0.0, 0.0, elementname[j]));
		f.write('ENDMDL\n');
	f.close();
	if verbose: print 'PDB file completed!';

#================================================

if __name__ == '__main__':

#	Code in progress:

	
	"""#==========================================================================	
	#	Kurtosis Sliding Window Computation

		#	coords = 3*numRes x numSamples
	window = config['kurtosis_window'];
	kurt = [];
	#dt = u.trajectory.dt; #	Time Delta btwn. frames
	h = (1/2.)*dt*window #	Half-Life of window (in nanoseconds), should be: dt*(window_len)/2??
	tao = (h/dt)/np.log(2);
	weights = np.arange(window);
	#alpha = 1 - np.exp(tao**-1);	#	From paper
	alpha = (1 - np.exp(tao**-1)) / (np.exp((-window - 1)*tao**-1) - 1);	#	reciprocal of summation of W(k)
	W = lambda k: alpha*np.exp(-(window-k)*tao**-1);
	weights = map(W, weights);
	
	log.debug( 'Sum of the weights: {0}'.format(np.sum(weights)));

	for i in range(window, coords.shape[1]):
		kurt.append(np.mean(kurtosis(coords[:,i-window:i].dot(np.diag(weights)), axis=1)));
	
	if val.save: np.save('savefiles/%s_kurtosis.npy' %(config['pname']), np.array(kurt));

	#	End Kurtosis
	#========================================================================== IN PROGRESS"""	

"""
	if val.graph:
		tmp = numpy.reshape(numpy.tile(avgCoords, num_coords), (num_coords,-1)).T;
		caDevsMD = coords - tmp;
		#print numpy.shape(caDevsMD); print caDevsMD[0];
		D = caDevsMD.flatten(); print numpy.shape(D);
		gm = numpy.mean(D); 
		gs = numpy.std(D);
		gK = scipy.stats.kurtosis(D,0,fisher=False);
		
		[n,s] = numpy.histogram(D, bins=51,normed=1);
		
		gp = numpy.exp(-(s-gm)**2/(2*gs*gs));
		gp = gp/numpy.sum(gp); print numpy.shape(gp);
		
		lo = 0; hi = len(s);
		
		fig = plt.figure();
		ax = fig.add_subplot(111);
		x = 0.5*(s[1:] + s[:-1]);
		#ax.semilogy(s[lo:hi], gp[lo:hi],'c-',linewidth=2);
		ax.hold(True); 
		ax.semilogy(x, n, 'k-', linewidth=2.0); ax.axis('tight');
		print 'fig1'
		plt.show();
	
	
		#print 'Mean: ', gm;
		print 'Std. dev: ', gs;
		print 'Kurtosis: ', gK;
		
		cc = coords[:,0]; print numpy.shape(cc);
		#print numpy.shape(coords[0:-1:3,0]), numpy.shape(coords[1:-1:3,0]), numpy.shape(coords[2:-1:3,0]);
		
		fig = plt.figure();
		ax = fig.add_subplot(111, projection='3d');
		ax.plot(cc[::dim], cc[1::dim], cc[2::dim]);
		print 'fig2'
		plt.show();
		
		numpy.save('cov_ca.npy', numpy.cov(coords));
		[pcas,pcab] = numpy.linalg.eigh(numpy.cov(coords));
		#numpy.save('pcab_ca.npy', pcab);
		print 'pcas: ', pcas
		si = numpy.argsort(-pcas.ravel()); print si;
		pcaTmp = pcas;
		pcas = numpy.diag(pcas);
		pcab = pcab[:,si];
	
		#fig = plt.figure();
		#ax = fig.add_subplot(111);
		#cs = ax.contourf(numpy.cov(coords));
		##ax.colorbar(cs); 
		#plt.show();
		
		fig = plt.figure();
		ax = fig.add_subplot(111, projection='3d');
		pcacoffs = numpy.dot(pcab.conj().T, caDevsMD);
		ax.scatter(pcacoffs[0,:], pcacoffs[1,:], pcacoffs[2,:], marker='o', c=[0.6,0.6,0.6]);
		print 'fig4';
		plt.show();

"""
