import numpy
import numpy as np
from numpy import *
import math
import scipy.stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from MDAnalysis.core.Timeseries import *
from MDAnalysis import *
import MDAnalysis
from jade import *
import argparse

#a is mem-mapped array, b is array in RAM we are adding to a.       
def mmap_concat(a,b):
	assert(a.shape[0] == b.shape[0]);
	c = np.memmap('dihedral_data.array', dtype='float64', mode='r+', shape=(a.shape[0],a.shape[1]+b.shape[1]), order='F')
	c[:, a.shape[1]: ] = b
	return c

def phisel(res):
#MDAnalysis' phi_selection requires a segid be present while this doesn't.
	return res.universe.selectAtoms('resid %d and name C' %(res.id-1) ) + res.N + res.CA + res.C

def psisel(res):
#MDAnalysis' phi_selection requires a segid be present while this doesn't. 
	return res.N + res.CA + res.C + res.universe.selectAtoms('resid %d and name N' %(res.id+1) )


#Main Code
num_traj = 1
rad_gyr = []

def qaa(num_traj, ica_dim, val):
	for i in range(num_traj):
		
		#	!Edit to your trajectory format!
		try:
			u = MDAnalysis.Universe("./traj-format_kbh/1KBH%i_ww.pdb" %(i+1), "./traj-format_kbh/1KBH_%i_50k.dcd" %(i+1), permissive=False);
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
			#print res
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
	
		#	Converts to nd-array and changes from [numRes,1,numSamples] to [numRes,numSamples]
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
	
	#	some set up for running JADE
	if val.debug: print 'fulldat: ', fulldat.shape
	Ncyc  = 1;
	subspace = ica_dim;
	lastEig = subspace; #	number of eigen-modes to be considered
	numOfIC = subspace; #	number of independent components to be resolved
	#	Runs jade	
	icajade = jadeR(fulldat, lastEig);

	if (val.save): np.save('icajadeKBH_30.npy', icajade) 
	if val.debug: print 'icajade shape: ', numpy.shape(icajade);
	#	Performs change of basis
	icacoffs = icajade.dot(fulldat)
	icacoffs = numpy.asarray(icacoffs); 
	
	if val.debug: print 'icacoffs shape: ', numpy.shape(icacoffs);
	if (val.save): numpy.save('icacoffsKBH_30.npy', icacoffs)
	
	if (val.graph):
		fig = plt.figure();
		ax = fig.add_subplot(111, projection='3d');
		ax.scatter(icacoffs[0,::10], icacoffs[1,::10], icacoffs[2,::10], marker='o', c=[0.6,0.6,0.6]); 
		print 'First 3-Dimensions of \'icacoffs\'';
		plt.show();
	
	return icacoffs;

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-g', action='store_true', dest='graph', default=False, help='Shows graphs.')
	parser.add_argument('-v', action='store_true', dest='verbose', default=False, help='Runs program verbosely.')
	parser.add_argument('-s', '--save', action='store_true', dest='save', default=False, help='Saves important matrices.')
	parser.add_argument('-d', '--debug', action='store_true', dest='debug', default=False, help='Prints debugging help.')

	values = parser.parse_args()
	if values.debug: values.verbose = True;
#	First is number of trajectories, second is dimensions you want jade to consider
	qaa(3, 30, values);
