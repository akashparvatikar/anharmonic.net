import numpy
import scipy.stats

def perResidueKurtosisTest(caDevsMDall, windowsize):
    n = caDevsMDall.shape[1] // windowsize;
    gK_pvalue = np.zeros((caDevsMDall.shape[0],n));
    gK_Zscore = [];
    gK_pvalue = [];
    for j in range(0,n):
        for i in range(0, caDevsMDall.shape[0]):
            cc = caDevsMDall[i,(windowsize*j):(windowsize*(j+1))-1:];
            kZscore, kpvalue = scipy.stats.kurtosistest(cc,0);
            
            gK_Zscore.append(kZscore);
            gK_pvalue.append(kpvalue);
            
    gK_Zscore = numpy.asarray(gK_Zscore)
    gK_Zscore = numpy.reshape(gK_Zscore, (caDevsMDall.shape[0],n));

    gK_pvalue = numpy.asarray(gK_pvalue)
    gK_pvalue = numpy.reshape(gK_pvalue, (caDevsMDall.shape[0],n));
    return gK_Zscore, gK_pvalue;
