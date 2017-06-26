import numpy
import scipy.stats; 

def getLongTails(caDevsMDall):
  
  D = caDevsMDall.flatten(); 
  [n,s] = numpy.histogram(D, bins=51,normed=1);
  gm = numpy.mean(D); 
  gs = numpy.std(D);
  gK = scipy.stats.kurtosis(D,0,fisher=False);
  print 'Overall kurtosis for system: ' + str(gK);
  gp = numpy.exp(-(s-gm)**2/(2*gs*gs));
  gp = gp/numpy.sum(gp); 

  print numpy.shape(gp);

  x = 0.5*(s[1:] + s[:-1]);

  return gK, x, n;
