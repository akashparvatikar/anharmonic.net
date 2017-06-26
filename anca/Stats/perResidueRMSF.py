import numpy

def perResidueRMSF(caDevsMDall, dim=None, Na=None):
  xx1 = numpy.std(caDevsMDall, 1);
  rmsf_1 = numpy.mean(xx1.reshape((dim, Na)),0);
  
  return rmsf_1;
