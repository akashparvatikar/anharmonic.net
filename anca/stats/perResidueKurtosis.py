import numpy
import scipy.stats

def perResidueKurtosis(caDevsMDall, Na=None):
  
  gK_Res = numpy.zeros((Na, 1)); 
  gK_Res_Zscore = numpy.zeros((Na,1));
  gK_Res_pvalue = numpy.zeros((Na,1));
  k = 0;
  for i in range(0, gK_Res.shape[0]):
      kX = scipy.stats.kurtosis(caDevsMDall[i,:],0,fisher=False);
      kY = scipy.stats.kurtosis(caDevsMDall[i+1,:],0,fisher=False);
      kZ = scipy.stats.kurtosis(caDevsMDall[i+2,:],0,fisher=False);
      kXZscore, kXpvalue = scipy.stats.kurtosistest(caDevsMDall[i,:],0);
      kYZscore, kYpvalue = scipy.stats.kurtosistest(caDevsMDall[i+1,:],0);
      kZZscore, kZpvalue = scipy.stats.kurtosistest(caDevsMDall[i+2,:],0);
      gK_Res[k] = numpy.mean([kX, kY, kZ],0);
      gK_Res_Zscore[k] = np.mean([kXZscore, kYZscore, kZZscore], 0);
      gK_Res_pvalue[k] = np.mean([kXpvalue, kYpvalue, kZpvalue], 0);
      i = i + 3;
      k = k + 1;
  
  return gK_Res, gK_Res_Zscore, gK_Res_pvalue;
