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


def average_coords(a):
	if len(a) == 1:
			return iterAlign.iterativeMeans(a[0], 0.001, 4);
	else:
		avgCoords = np.zeros((3,69))
		for i in range(len(a)):
			tmpa, avgCoords[:,i], tmpc, tmpd = iterAlign.iterativeMeans(cacoords[i], 0.001, 4);
		return average_coords(avgCoords, 0.001, 4);       
"""
u = MDAnalysis.Universe('./ubq_1111.pdb', './UBQ_500ns.dcd', permissive=False);
#u = MDAnalysis.Universe('../../tmp/2V93_1.pdb', '../../tmp/2V93.dcd', permissive=False);
#ca = u.selectAtoms('name CA');
ca = u.selectAtoms('backbone');
cacoords = []; frames = [];

for ts in u.trajectory:
	f = ca.coordinates();
	cacoords.append(f.T);
	frames.append(ts.frame);

print numpy.shape(cacoords);


dim = 3; Na = 69;
iterAlign = IterativeMeansAlign();
tmp = []; #tmp will store the LOTS of atomcoords

[a, b, c, d] = iterAlign.iterativeMeans(cacoords, 0.001, 4);
print array(b[-1]).shape
asdf
"""
count = 0
dim = 3; Na = 276;
iterAlign = IterativeMeansAlign();
tmp = []; #tmp will store the LOTS of atomcoords
num_coords = 1
for i in range(num_coords):
	if i < 10:
		tm = '00';
	elif i < 100:
		tm = '0';
	else:
		tm = '';
	u = MDAnalysis.Universe("lacie/UBQ/native-1/pnas2013-native-1-protein/protein.pdb", "lacie/UBQ/native-1/pnas2013-native-1-protein/pnas2013-native-1-protein-%s.dcd" %(tm+str(i)), permissive=False);
	atom = u.selectAtoms('backbone');
	print atom[4:8];

	atomcoords = []; frames = [];                    
	numresidues = atom.numberOfResidues()

	phidat = TimeseriesCollection()
	psidat = TimeseriesCollection()

	#Adds each (wanted) residues phi/psi angles to their respective timeseries collections.
	for res in range(1,70):
		print "Processing residue %d" % res
		#  selection of the atoms involved for the phi for resid '%d' %res
		## selectAtoms("atom 4AKE %d C"%(res-1), "atom 4AKE %d N"%res, "atom %d 4AKE CA"%res, "atom 4AKE %d C" % res)
		phi_sel = u.residues[res].phi_selection()
		if res % 20 == 0: print phi_sel[0], phi_sel[1], phi_sel[2], phi_sel[3], '\n'
		#  selection of the atoms involved for the psi for resid '%d' %res
		psi_sel = u.residues[res].psi_selection()
		if res % 20 == 0: print psi_sel[0], psi_sel[1], psi_sel[2], psi_sel[3]
		#print array(u.trajectory).shape
		#collection.addTimeseries(Timeseries.Dihedral(phi_sel))
		#collection.addTimeseries(Timeseries.Dihedral(psi_sel))
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
	print phidat.shape
	
	dihedral_dat = np.zeros((69,4,10000))
	#Data stored as | sin(phi) | cos(phi) | sin(psi) | cos(psi) |
	dihedral_dat[:,0,:] = np.sin(phidat)
	dihedral_dat[:,1,:] = np.cos(phidat)
	dihedral_dat[:,2,:] = np.sin(psidat)
	dihedral_dat[:,3,:] = np.cos(psidat)

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

# some set up for running JADE
Ncyc  = 1;
subspace = 30;
lastEig = subspace; # number of eigen-modes to be considered
numOfIC = subspace; # number of independent components to be resolved

icajade = jadeR(coords, lastEig); 
print numpy.shape(icajade);
icacoffs = numpy.dot(icajade, caDevsMD);
icacoffs = numpy.asarray(icacoffs); 
print 'icacoffs: ', numpy.shape(icacoffs);
numpy.save('ica.npy', icacoffs)
fig = plt.figure();
ax = fig.add_subplot(111, projection='3d');
ax.scatter(icacoffs[0,:], icacoffs[1,:], icacoffs[2,:], marker='o', c=[0.6,0.6,0.6]); 
print 'fig5';
plt.show();
