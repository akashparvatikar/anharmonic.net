import math
import numpy
import numpy.linalg
import MDAnalysis

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class KabschAlign(object):
	
	def __init__(self):
		"""
		Constructor
		"""

	def kabsch(self, toXYZ, fromXYZ, counter):
		"""
		Input is a 3 x N array of coordinates.
		"""
		# This file has been edited to produce identical results as the original matlab implementation.
		len1 = numpy.shape(fromXYZ);
		len2 = numpy.shape(toXYZ);
	
		if not(len1[1] == len2[1]):
			print 'KABSCH: unequal array sizes';
			return;
		
		m1 = numpy.mean(fromXYZ, 1).reshape((len1[0],1)); # print numpy.shape(m1);
		m2 = numpy.mean(toXYZ, 1).reshape((len2[0],1)); 

		tmp1 = numpy.tile(m1,len1[1]);
		tmp2 = numpy.tile(m1,len2[1]);

		assert numpy.allclose(tmp1, tmp2);
		assert tmp1.shape == fromXYZ.shape;
		assert tmp2.shape == toXYZ.shape;

		t1 = fromXYZ - tmp1;
		t2 = toXYZ - tmp2;

		[u, s, wh] = numpy.linalg.svd(numpy.dot(t2,t1.T));
		w = wh.T;

		R = numpy.dot(numpy.dot(u,[[1, 0, 0],[0, 1, 0],[0, 0, numpy.linalg.det(numpy.dot(u,w.T))]]), w.T); 
  		T = m2 - numpy.dot(R,m1);

		tmp3 = numpy.reshape(numpy.tile(T,(len2[1])),(len1[0],len1[1]));
  		err = toXYZ - numpy.dot(R,fromXYZ) - tmp3; 
  			
		#eRMSD = math.sqrt(sum(sum((numpy.dot(err,err.T))))/len2[1]); 
		eRMSD = math.sqrt(sum(sum(err**2))/len2[1]); 
		return (R, T, eRMSD, err.T);
	
	def wKabschDriver(self, toXYZ, fromXYZ, sMed=1.5, maxIter=20):
		scaleMed = sMed;
		weights = numpy.ones((1, numpy.shape(toXYZ)[1])); print 'weights: ', numpy.shape(weights);
		flagOut = 0;
		Rc = []; Tc = []; sigc = [];
		for itr in range(0, maxIter):
			[R, T, eRMSD, err] = self.wKabsch(toXYZ, fromXYZ, weights);
			Rc.append(R);
			Tc.append(T);
			tmp1 = numpy.reshape(numpy.tile(T, (numpy.shape(toXYZ[1]))), (numpy.shape(toXYZ)[0],numpy.shape(toXYZ)[1]));
			deltaR = numpy.array( numpy.dot(R, fromXYZ) + tmp1 - toXYZ ); print 'deltaR shape: ', numpy.shape(deltaR);
			print deltaR;
			numpy.save('deltaR.npy', deltaR);
			nDeltaR = numpy.sqrt(numpy.sum(deltaR**2, axis = 0)); print 'nDeltaR shape:', numpy.shape(nDeltaR);
			sig = scaleMed*numpy.median(nDeltaR);
			sigc.append(sig);
			weights = (sig**2)/((sig**2 + nDeltaR**2)**2); print numpy.shape(weights);
		return ( R, T, eRMSD, err);
			
	def wKabsch(self, toXYZ, fromXYZ, weights):
		len1 = numpy.shape(fromXYZ); #print 'len1: ', len1;
		len2 = numpy.shape(toXYZ); #print 'len2: ', len2;
		
		if not(len1[1] == len2[1]):
			print 'wKABSCH: unequal array sizes';
			return;
		
		if not (numpy.shape(weights)[0]==1):
			print 'am here:'
			weights = weights.T;
			
		dw = numpy.tile(weights, (3,1)); #print 'dw shape:', numpy.shape(dw);
		wFromXYZ = dw * fromXYZ; #print 'wFromXYZ shape: ', numpy.shape(wFromXYZ);
		wToXYZ = dw * toXYZ; # print 'wToXYZ shape: ', numpy.shape(wToXYZ);
		
		m1 = numpy.sum(wFromXYZ, 1) / numpy.sum(weights); print numpy.shape(m1);
		m2 = numpy.sum(wToXYZ, 1) / numpy.sum(weights); print numpy.shape(m2);
		
		tmp1 = numpy.reshape(numpy.tile(m1,(len1[1])), (len1[0],len1[1]));
		tmp2 = numpy.reshape(numpy.tile(m2,(len2[1])), (len2[0],len2[1])); 
		t1 = numpy.reshape(fromXYZ - tmp1, (len1[0], len1[1])); print 't1 shape: ', numpy.shape(t1);
		t2 = numpy.reshape(toXYZ - tmp2, (len2[0],len2[1]));
		
		aa = numpy.zeros((3,3));
		for i in range(0, numpy.shape(t1)[1]):
			tmp = numpy.outer(t2[:,i],t1[:,i]); print 'tmp shape: ', numpy.shape(tmp);
			aa = aa + numpy.multiply(weights[0,i], tmp);
		aa = aa/numpy.sum(weights);
		
		[u,s,wh] = numpy.linalg.svd(aa);
		w = wh.T;

		R = numpy.dot(numpy.dot(u,[[1, 0, 0],[0, 1, 0],[0, 0, numpy.linalg.det(numpy.dot(u,w.T))]]), w.T); 
  		T = m2 - numpy.dot(R,m1); 
		#print numpy.shape(m2), numpy.shape(R), numpy.shape(m1); 

		tmp3 = numpy.reshape(numpy.tile(T,(len2[1])),(len1[0],len1[1]));
  		err = toXYZ - numpy.dot(R,fromXYZ) - tmp3; 
  			
		#eRMSD = math.sqrt(sum(sum((numpy.dot(err,err.T))))/len2[1]); 
		eRMSD = math.sqrt(sum(sum(err**2))/len2[1]); 
		return (R, T, eRMSD, err.T);
		
		

if __name__ == '__main__':
	u = MDAnalysis.Universe('../data/ubq_1111.pdb', '../data/UBQ_500ns.dcd', permissive=False);
	ca = u.selectAtoms('name CA');
	f1 = ca.coordinates();
	ka = KabschAlign(); RMS = []; frames = [];
	for ts in u.trajectory:
		f2 = ca.coordinates();
		[R, T, eRMSD, err] = ka.kabsch(f2.T, f1.T);
		RMS.append(eRMSD);
		frames.append(ts.frame);

	rmsRef = [];
	f = open('../data/rms.log', 'r');
	for line in f.readlines():
		l = line.strip().split();
		rmsRef.append(float(l[1]));
	f.close();

	print numpy.corrcoef(rmsRef, RMS);

	fig = plt.figure();
	ax = fig.add_subplot(111);
	ax.plot(frames, RMS, 'r-', frames, rmsRef, 'b--', linestyle='solid', linewidth=2.0);

	plt.show();	
