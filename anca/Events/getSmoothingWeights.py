import numpy

def getSmoothingWeights(dt, windowsize, halflife):
    tau = (halflife/dt)/ numpy.log(2) 
    alpha = 1 - numpy.exp(-1/tau)
    s = 0;
    wt = numpy.zeros((windowsize,1));
    for i in range (0,windowsize):
        wt[i] = (alpha * numpy.exp(-(windowsize-i)/tau))/0.96499303
        s = s + wt[i];
    return wt;
