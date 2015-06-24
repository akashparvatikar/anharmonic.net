import numpy
import numpy as np
from numpy import *
import math
import scipy.stats
import matplotlib.pyplot as plt
from MDAnalysis.core.Timeseries import *
from MDAnalysis import *
from KabschAlign import *
from IterativeMeansAlign import *
from jade import *

#a is mem-mapped array, b is array in RAM we are adding to a.       
def mmap_concat(a,b):
	assert(a.shape[0] == b.shape[0]);
	c = np.memmap('dihedral_data.array', dtype='float64', mode='r+', shape=(a.shape[0],a.shape[1]+b.shape[1]), order='F')
	c[:, a.shape[1]: ] = b
	return c

def phisel(res):
#MDAnalysis' phi_selection requires a segid be present, this doesn't.  use if MDanalysis fails.
	return res.universe.selectAtoms('resid %d and name C' %(res.id-1) ) + res.N + res.CA + res.C

def psisel(res):
#MDAnalysis' phi_selection requires a segid be present, this doesn't.  use if MDanalysis fails.
	return res.N + res.CA + res.C + res.universe.selectAtoms('resid %d and name N' %(res.id+1) )


#Main Code
num_traj = 1
rad_gyr = []

def main(num_traj, ica_dim):
	for i in range(num_traj):
		
		#	!Edit to your trajectory format!
		try:
			u = MDAnalysis.Universe("lacie-kbhdata/1KBHww.pdb", "lacie-kbhdata/1KBH_%03i_50k.dcd" %(i+1), permissive=False);
		except:
			print "You must edit \'dQAA.py\' to fit your trajectory format!";
			exit();
	
		atom = u.selectAtoms('backbone');
	
		phidat = TimeseriesCollection()
		psidat = TimeseriesCollection()
	
		#	Adds each (wanted) residues phi/psi angles to their respective timeseries collections.
		print '---Processing Trajectory %i---' %(i+1)
		numres = 0
		trajlen = len(u.trajectory)
	
		#	Tailor following for-loop to iterate through your residues of interest
		for res in range(1,atom.numberOfResidues()-1):
			print res
			#	selection of the atoms involved for the phi for resid '%d' %res
			phi_sel = phisel(u.residues[res])
			#	selection of the atoms involved for the psi for resid '%d' %res
			psi_sel = psisel(u.residues[res])
	
			phidat.addTimeseries(Timeseries.Dihedral(phi_sel))
			psidat.addTimeseries(Timeseries.Dihedral(psi_sel))
			numres = numres + 1
	
		#	Computes along 10K timesteps
		phidat.compute(u.trajectory)
		psidat.compute(u.trajectory)
	
		#	Converts to nd-array and changes from [69,1,10K] to [69,10K]
		phidat =  array(phidat)
		phidat = phidat.reshape(phidat.shape[0],phidat.shape[2])
		psidat =  array(psidat)
		psidat = psidat.reshape(psidat.shape[0],psidat.shape[2])
		
		dihedral_dat = np.zeros((numres*4,trajlen))
		#	Data stored as | sin(phi) | cos(phi) | sin(psi) | cos(psi) |
		dihedral_dat[0::4,:] = np.sin(phidat)
		dihedral_dat[1::4,:] = np.cos(phidat)
		dihedral_dat[2::4,:] = np.sin(psidat)
		dihedral_dat[3::4,:] = np.cos(psidat)
		
		if i == 0:
			fulldat = np.memmap('dihedral_data.array', dtype='float64', mode='w+', shape=(numres*4, trajlen))
			fulldat[:,:] = dihedral_dat
		else:
			fulldat = mmap_concat(fulldat, dihedral_dat);
		for ts in u.trajectory:
			rad_gyr.append( atom.radiusOfGyration() )
	
	#	Original AnharmonicHisto code showing kurtosis, pca plots, etc...
	"""
	print array(tmp).shape
	itr = []; avgCoords = []; eRMSD = []; newCoords = [];
	
	for i in range(num_coords):
		[a, b, c, d] = iterAlign.iterativeMeans(array(tmp)[i,:,:,:], 0.001, 4);
		itr.append(a);
		avgCoords.append(b[-1]);
		eRMSD.append(c);
		newCoords.append(d);
	
	print array(avgCoords).shape
	
	#Final averaging
	[itr, avgCoords, eRMSD, newCoords] = iterAlign.iterativeMeans(array(avgCoords), 0.001, 4);
	
	print 'eRMSD shape: ', numpy.shape(eRMSD);
	print 'newC shape: ', newCoords.shape;
	print 'len: ', len(newCoords)
	coords = numpy.reshape(newCoords, (len(newCoords), dim*Na)).T;
	
	print 'coords: ', numpy.shape(coords); 
	
	avgCoords = numpy.mean(coords, 1); print avgCoords;
	
	print 'avgCoords: ', numpy.shape(avgCoords);
	
	tmp = numpy.reshape(numpy.tile(avgCoords, num_coords), (num_coords,dim*Na)).T;
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
	cc = numpy.reshape(cc, (dim,Na));
	#print numpy.shape(coords[0:-1:3,0]), numpy.shape(coords[1:-1:3,0]), numpy.shape(coords[2:-1:3,0]);
	
	fig = plt.figure();
	ax = fig.add_subplot(111, projection='3d');
	ax.plot(cc[0,:], cc[1,:], cc[2,:]);
	print 'fig2'
	plt.show();
	
	print numpy.shape(numpy.cov(coords));
	numpy.save('cov_ca.npy', numpy.cov(coords));
	[pcas,pcab] = numpy.linalg.eigh(numpy.cov(coords));
	#numpy.save('pcab_ca.npy', pcab);
	print pcas.shape, pcab.shape
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
	ax = fig.add_subplot(111);
	y = numpy.cumsum(pcaTmp.ravel()/numpy.sum(pcaTmp.ravel()));
	ax.plot(y);
	print 'fig3';
	plt.show();
	
	fig = plt.figure();
	ax = fig.add_subplot(111, projection='3d');
	pcacoffs = numpy.dot(pcab.conj().T, caDevsMD);
	print numpy.shape(pcacoffs);
	ax.scatter(pcacoffs[0,:], pcacoffs[1,:], pcacoffs[2,:], marker='o', c=[0.6,0.6,0.6]);
	print 'fig4';
	plt.show();
	"""
	
	#	some set up for running JADE
	print 'fulldat: ', fulldat.shape
	Ncyc  = 1;
	subspace = ica_dim;
	lastEig = subspace; #	number of eigen-modes to be considered
	numOfIC = subspace; #	number of independent components to be resolved
	
	plt.plot(range(trajlen),rad_gyr[:], 'r--', lw=2)
	plt.show()
	
	icajade = jadeR(fulldat, lastEig);
	np.save('icajade.npy', icajade) 
	print 'icajade shape: ', numpy.shape(icajade);
	icacoffs = icajade.dot(fulldat)
	icacoffs = numpy.asarray(icacoffs); 
	print 'icacoffs shape: ', numpy.shape(icacoffs);
	numpy.save('icacoffs.npy', icacoffs)
	
	fig = plt.figure();
	ax = fig.add_subplot(111, projection='3d');
	ax.scatter(icacoffs[0,:], icacoffs[1,:], icacoffs[2,:], marker='o', c=[0.6,0.6,0.6]); 
	print 'First 3-Dimensions of \'icacoffs\'';
	plt.show();

if __name__ == '__main__':
	main(1, 40);
