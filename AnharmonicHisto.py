import numpy
import math
import scipy.stats

from KabschAlign import *
from IterativeMeansAlign import *

from MDAnalysis import *

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
dim = 3; Na = 69;
iterAlign = IterativeMeansAlign();
tmp = []; #tmp will store the LOTS of atomcoords
num_coords = 2
for i in range(num_coords):
	if i < 10:
		tm = '00';
	elif i < 100:
		tm = '0';
	else:
		tm = '';
	u = MDAnalysis.Universe("lacie/UBQ/native-1/pnas2013-native-1-protein/protein.pdb", "lacie/UBQ/native-1/pnas2013-native-1-protein/pnas2013-native-1-protein-%s.dcd" %(tm+str(i)), permissive=False);
	atom = u.selectAtoms('name N or C');
	atomcoords = []; frames = [];                    

	for ts in u.trajectory:
		f = atom.coordinates();
		#print f.shape
		atomcoords.append(f.T);
		frames.append(ts.frame);
	tmp.append(atomcoords);
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

tmp = numpy.reshape(numpy.tile(avgCoords, 10000), (10000,dim*Na)).T;
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
plt.show();

print numpy.shape(numpy.cov(coords));
[pcas,pcab] = numpy.linalg.eig(numpy.cov(coords));
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
plt.show();

fig = plt.figure();
ax = fig.add_subplot(111, projection='3d');
pcacoffs = numpy.dot(pcab.conj().T, caDevsMD);
print numpy.shape(pcacoffs);
ax.scatter(pcacoffs[0,:], pcacoffs[1,:], pcacoffs[2,:], marker='o', c=[0.6,0.6,0.6]);
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
plt.show();
