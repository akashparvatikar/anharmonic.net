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

#a is mem-mapped array, b is array in RAM we are adding to a.       
def mmap_concat(a,b):
	c = np.memmap('dihedral_data.array', dtype='float32', mode='r+', shape=(276,a.shape[1]+b.shape[1]), order='F')
	c[:, a.shape[1]: ] = b
	return c

#Main Code
num_traj = 1
rad_gyr = []
for i in range(num_traj):
	if i < 10:
		tm = '00';
	elif i < 100:
		tm = '0';
	else:
		tm = '';
	u = MDAnalysis.Universe("lacie/UBQ/native-1/pnas2013-native-1-protein/protein.pdb", "lacie/UBQ/native-1/pnas2013-native-1-protein/pnas2013-native-1-protein-%s.dcd" %(tm+str(i)), permissive=False);
	atom = u.selectAtoms('backbone');

	phidat = TimeseriesCollection()
	psidat = TimeseriesCollection()

	#Adds each (wanted) residues phi/psi angles to their respective timeseries collections.
	print '---Processing Trajectory %d' %(i+1)
	print '---Intentionally excluding first residue and all residues after the 70th---';
	
	for res in range(1,70):
		#  selection of the atoms involved for the phi for resid '%d' %res
		phi_sel = u.residues[res].phi_selection()
		#  selection of the atoms involved for the psi for resid '%d' %res
		psi_sel = u.residues[res].psi_selection()

		phidat.addTimeseries(Timeseries.Dihedral(phi_sel))
		psidat.addTimeseries(Timeseries.Dihedral(psi_sel))

	#Computes along 10K timesteps (I think...)
	phidat.compute(u.trajectory)
	psidat.compute(u.trajectory)

	#Converts to nd-array and changes from [69,1,10K] to [69,10K]
	phidat =  array(phidat)
	phidat = phidat.reshape(phidat.shape[0],phidat.shape[2])
	psidat =  array(psidat)
	psidat = psidat.reshape(psidat.shape[0],psidat.shape[2])
	
	dihedral_dat = np.zeros((69,4,10000))
	#Data stored as | sin(phi) | cos(phi) | sin(psi) | cos(psi) |
	dihedral_dat[:,0,:] = np.sin(phidat)
	dihedral_dat[:,1,:] = np.cos(phidat)
	dihedral_dat[:,2,:] = np.sin(psidat)
	dihedral_dat[:,3,:] = np.cos(psidat)
	dihedral_dat = dihedral_dat.reshape(-1,10000)
	
	if i == 0:
		fulldat = np.memmap('dihedral_data.array', dtype='float32', mode='w+', shape=(276, 10000))
		fulldat[:,:] = dihedral_dat
	else:
		fulldat = mmap_concat(fulldat, dihedral_dat);
	for ts in u.trajectory:
		rad_gyr.append( atom.radiusOfGyration() )
#Not sure if this is needed, decided to leave in in case some stats could be made off phi/psi angles
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

# some set up for running JADE
print 'fulldat: ', fulldat.shape
Ncyc  = 1;
subspace = 20;
lastEig = subspace; # number of eigen-modes to be considered
numOfIC = subspace; # number of independent components to be resolved

plt.plot(range(10000),rad_gyr[:], 'r--', lw=2)
plt.show()

icajade = jadeR(fulldat, lastEig); 
print 'icajade: ', numpy.shape(icajade);
icacoffs = icajade.dot(fulldat)
icacoffs = numpy.asarray(icacoffs); 
print 'icacoffs: ', numpy.shape(icacoffs);
#numpy.save('ica.npy', icacoffs)
fig = plt.figure();
ax = fig.add_subplot(111, projection='3d');
ax.scatter(icacoffs[0,:], icacoffs[1,:], icacoffs[2,:], marker='o', c=[0.6,0.6,0.6]); 
print 'fig5';
plt.show();
