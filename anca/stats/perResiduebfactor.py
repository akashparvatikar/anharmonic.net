import numpy

def perResiduebfactor(caDevsMDall, dim=None, Na=None):
  
  mm = caDevsMDall.mean(1);
  md = caDevsMDall.std(1);
  cntArray = numpy.zeros((174,1));
  for i in range(0, caDevsMDall.shape[0]):
      y = caDevsMDall[i,:]
      b = filter(lambda x: (x >= (mm[i] + 2.5*md[i]) or x <= (mm[i] - 2.5*md[i])), y); 
      cntArray[i] = len(b);

  cntArray = cntArray.reshape((dim,Na));
  bfactor = numpy.zeros((Na,1));
  for i in range(0, Na):
      bfactor[i] = float(sum(cntArray[:,i]))*100/Nc;
  
  return bfactor;
