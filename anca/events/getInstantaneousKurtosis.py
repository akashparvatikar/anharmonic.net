import numpy
import scipy.stats

def getInstantaneousKurtosis(caDevsMDall, dt, windowsize, halflife, wt, perRes=True, smooth=False):
    kresVals = numpy.zeros((caDevsMDall.shape[0], caDevsMDall.shape[1]));
    kvals = numpy.zeros((caDevsMDall.shape[1],1));
    kresZscorevals = numpy.zeros((caDevsMDall.shape[0],caDevsMDall.shape[1]));
    krespvals = numpy.zeros((caDevsMDall.shape[0],caDevsMDall.shape[1]));
    
    kZscorevals =  numpy.zeros((caDevsMDall.shape[1],1));
    kpvals = numpy.zeros((caDevsMDall.shape[1],1));
    for i in range(0, caDevsMDall.shape[1]):
        cc = caDevsMDall[:, i:i+windowsize];
        kresVals[:, i] = scipy.stats.kurtosis(cc, axis=1, fisher=False);
        kvals[i] = scipy.stats.kurtosis(cc, axis=1, fisher=False).mean();
        kresZscorevals[:,i],krespvals[:,i] = scipy.stats.kurtosistest(cc, axis=1, nan_policy='omit');
    
    kZscorevals = numpy.mean(kresZscorevals, axis = 0);    
    kpvals = numpy.mean(krespvals, axis = 0);
    kZscore = kZscorevals.tolist();
    kpvalue = kpvals.tolist();
    
    c = numpy.zeros(caDevsMDall.shape[1]-windowsize);
    if not perRes and smooth:
        val = numpy.zeros((windowsize,1));
        for i in range (0, caDevsMDall.shape[1]-windowsize):
            a = wt[0:windowsize];
            a = a.reshape(windowsize);
            b = kvals[i + windowsize:i:-1];
            b = b.reshape(windowsize);
            c[i] = numpy.dot(a,b);
        return c, kvals, kZscore, kpvalue;
    elif perRes:
        return kresVals, kvals, kresZscorevals, krespvals;
    else:
        return kvals;
