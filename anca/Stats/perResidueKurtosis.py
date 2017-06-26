import numpy

def perResidueKurtosis(caDevsMDall, Na=None):
  
  gK_Res = numpy.zeros((Na, 1)); 
  k = 0;
  for i in range(0, gK_Res.shape[0]):
      kX = scipy.stats.kurtosis(caDevsMDall[i,:],0,fisher=False);
      kY = scipy.stats.kurtosis(caDevsMDall[i+1,:],0,fisher=False);
      kZ = scipy.stats.kurtosis(caDevsMDall[i+2,:],0,fisher=False);
      gK_Res[k] = numpy.mean([kX, kY, kZ],0);
      i = i + 3;
      k = k + 1;
  
  return gK_Res;
