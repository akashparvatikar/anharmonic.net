import numpy
import numpy as np
import math
import scipy.stats
import matplotlib.pyplot as plt
from KabschAlign import *
from IterativeMeansAlign import *
from MDAnalysis.core.Timeseries import *
from MDAnalysis import *
from numpy import *
from jade import *
import timing
import argparse

#a is mem-mapped array, b is array in RAM we are adding to a.
def mmap_concat(a,b):
	assert(a.shape[1] == b.shape[1] and a.shape[2] == b.shape[2]);
	c = np.memmap('coord_data.array', dtype='float64', mode='r+', shape=(a.shape[0]+b.shape[0],a.shape[1],a.shape[2]), order='F')
	c[a.shape[0]:, :, : ] = b
	return c

#	Main  Code
#	||		||
#	\/		\/

#================================================
def qaa(config, val):
	iterAlign = IterativeMeansAlign();
	itr = []; avgCoords = []; eRMSD = [];
	start_traj = config['startTraj'];
	num_traj = config['numOfTraj'];
	dim = 3;
	startRes = config['startRes'];
	numRes = config['numRes'];
	for i in range(start_traj,num_traj):
		#	!Edit to your trajectory format!
		try:
			u = MDAnalysis.Universe("ubq/protein.pdb", "ubq/pnas2013-native-1-protein-%03i.dcd" %(i), permissive=False);
		except:
			raise ImportError('You must edit \'cQAA.py\' to fit your trajectory format!');
			exit();

		atom = u.selectAtoms('name CA');

		if val.verbose: timing.log('Processing Trajectory %i...' %(i+1))
		counter = 0;
		cacoords = []; frames = [];
		for ts in u.trajectory:
			if (counter % config['slice_val'] == 0):
				f = atom.coordinates();
				cacoords.append(f.T);
				frames.append(ts.frame);
			counter = counter + 1;
		
		if numRes == -1:
			numRes = atom.numberOfResidues();

		if atom.numberOfResidues() == numRes:
			[a, b, c, d] = iterAlign.iterativeMeans(array(cacoords)[:,:,:], 0.150, 4, val.verbose);
		else:
			[a, b, c, d] = iterAlign.iterativeMeans(array(cacoords)[:,:,startRes:startRes+numRes], 0.150, 4, val.verbose);				
		itr.append(a);
		avgCoords.append(b[-1]);
		eRMSD.append(c);
		if ( i == start_traj):
			fulldat = np.memmap('coord_data.array', dtype='float64', mode='w+', shape=(d.shape[0], d.shape[1], d.shape[2]));
			fulldat[:,:,:] = d;
		else: fulldat = mmap_concat(fulldat, d);

	#	Final averaging
	num_coords = fulldat.shape[0];
	dim = fulldat.shape[1];
	num_atoms = fulldat.shape[2];
	if val.debug: print 'num_coords: ', num_coords;
	if num_traj > 1:
		[itr, avgCoords, eRMSD, fulldat[:,:,:] ] = iterAlign.iterativeMeans(fulldat, 0.150, 4, val.verbose);	

	if val.debug: print 'eRMSD shape: ', numpy.shape(eRMSD);
	if val.save: np.save('%s_eRMSD.npy' %(config['pname']), eRMSD );

	#	Reshaping of coords
	coords = np.memmap('cqaa.array', dtype='float64', mode='w+', shape=(fulldat.shape[1]*fulldat.shape[2], fulldat.shape[0]));
	coords[:,:] = fulldat.reshape((fulldat.shape[0],-1), order='F').T

	jade_calc(coords, avgCoords, num_coords);

#================================================
def minqaa(config, val, fulldat):
	iterAlign = IterativeMeansAlign();
	itr = []; avgCoords = []; eRMSD = [];
	dim = 3;

	#	Final averaging
	assert(len(fulldat.shape) == 3);
	num_coords = fulldat.shape[0];
	dim = fulldat.shape[1];
	num_atoms = fulldat.shape[2];

	[itr, avgCoords, eRMSD, fulldat ] = iterAlign.iterativeMeans(fulldat, 0.150, 4, val.verbose);	
	
	if val.debug: print 'eRMSD shape: ', numpy.shape(eRMSD);
	if val.save: np.save('%s_eRMSD.npy' %(config['pname']), eRMSD );
	#	Reshaping of coords
	coords = fulldat.reshape((fulldat.shape[0],-1), order='F').T

	jade_calc(coords, avgCoords, num_coords);

#================================================
def jade_calc(coords, avgCoords, num_coords):

	if val.debug: print 'coords: ', numpy.shape(coords); 
	
	avgCoords = numpy.mean(coords, 1); 
	if val.debug: print avgCoords;
	
	if val.debug: print 'avgCoords: ', numpy.shape(avgCoords);
	
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
		if val.verbose: print 'Std. dev: ', gs;
		if val.verbose: print 'Kurtosis: ', gK;
		
		cc = coords[:,0]; print numpy.shape(cc);
		#print numpy.shape(coords[0:-1:3,0]), numpy.shape(coords[1:-1:3,0]), numpy.shape(coords[2:-1:3,0]);
		
		fig = plt.figure();
		ax = fig.add_subplot(111, projection='3d');
		ax.plot(cc[::dim], cc[1::dim], cc[2::dim]);
		print 'fig2'
		plt.show();
		
		if val.debug: print numpy.shape(numpy.cov(coords));
		numpy.save('cov_ca.npy', numpy.cov(coords));
		[pcas,pcab] = numpy.linalg.eigh(numpy.cov(coords));
		#numpy.save('pcab_ca.npy', pcab);
		if val.debug: print pcas.shape, pcab.shape
		if val.verbose: print 'pcas: ', pcas
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
		if val.debug: print numpy.shape(pcacoffs);
		ax.scatter(pcacoffs[0,:], pcacoffs[1,:], pcacoffs[2,:], marker='o', c=[0.6,0.6,0.6]);
		print 'fig4';
		plt.show();
	
	if val.setup:
		[pcas,pcab] = numpy.linalg.eig(numpy.cov(coords));
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
		if val.verbose: print('Cov. Matrix spectrum cum. sum.');
		plt.show();
		a = input('Enter desired ICA dimension (enter -1 for default): ');
		if (a > 0):
			config['icadim'] = a;
	
	if val.debug and val.save:
		np.save('coords_%s.npy' %(config['pname']) , coords);	

	# some set up for running JADE
	subspace = config['icadim'];
	lastEig = subspace; # number of eigen-modes to be considered
	numOfIC = subspace; # number of independent components to be resolved
	
	#	Performs jade and saves if main
	icajade = jadeR(coords, lastEig);
	if (val.save) and __name__ == '__main__': np.save('icajade_%s_%i.npy' %(config['pname'], config['icadim']), icajade) 
	if val.debug: print 'icajade: ', numpy.shape(icajade);

	#	Performs change of basis
	icacoffs = icajade.dot(coords)
	icacoffs = numpy.asarray(icacoffs);
	if val.debug: print 'icacoffs: ', numpy.shape(icacoffs);
	if (val.save) and __name__ == '__main__': np.save('icacoffs_%s_%i.npy' %(config['pname'], config['icadim']), icacoffs) 
	
	if val.graph:	
		fig = plt.figure();
		ax = fig.add_subplot(111, projection='3d');
		ax.scatter(icacoffs[0,:], icacoffs[1,:], icacoffs[2,:], marker='o', c=[0.6,0.6,0.6]); 
		print 'First 3 Dimensions of Icacoffs';
		plt.show();

	#	Saves array's to a dict and returns
	icamat = {};
	icamat['icajade'] = icajade;
	icamat['icacoffs'] = icacoffs;
	return icamat;

#================================================
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-g', action='store_true', dest='graph', default=False, help='Shows graphs.')
	parser.add_argument('-v', action='store_true', dest='verbose', default=False, help='Runs program verbosely.')
	parser.add_argument('-s', '--save', action='store_true', dest='save', default=False, help='Saves important matrices.')
	parser.add_argument('-d', '--debug', action='store_true', dest='debug', default=False, help='Prints debugging help.')
	parser.add_argument('--setup', action='store_true', dest='setup', default=False, help='Runs setup calculations: Cum. Sum. of cov. spectrum\nand unit radius neighbor search.')
	parser.add_argument('-i', '--input', type=str, dest='coord_in', default='null', help='Allows direct inclusion of an array of coordinates. Input as [numRes, 3, numSamp].')

	values = parser.parse_args()
	if values.debug: values.verbose = True;

#	Config settings -- only here if cQAA is called directly from python
	config = {};
	config['numOfTraj'] = 1;
	config['startTraj'] = 0;
	config['icadim'] = 40;
	config['pname'] = 'pname';	#	Edit to fit your protein name
	config['startRes'] = 0;
	config['numRes']=-1;
	config['slice_val'] = 1;
	if (values.coord_in == 'null'):
		qaa(config, values);
	else:
		minqaa(config, values, np.load(values.coord_in)); 

